import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from .basic_var import AdaLNBeforeHead, AdaLNSelfAttn
from .helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from accelerate.logging import get_logger

logger = get_logger(__name__)

def create_attn_bias_for_masking(patch_nums, dynamic_length, context_length, device, c_patch_nums=None):
    patch_nums_d = patch_nums * dynamic_length
    L = sum(pn ** 2 for pn in patch_nums_d)
    dyn_sf = torch.cat([torch.full((pn * pn,), i) for i, pn in enumerate(patch_nums_d)]).view(1, L, 1).to(device)
    
    if c_patch_nums is not None:
        ctx_tokens_per_frame = c_patch_nums[-1] ** 2
    else:
        ctx_tokens_per_frame = patch_nums[-1] ** 2
        
    total_ctx_tokens = context_length * ctx_tokens_per_frame
    context = - torch.ones(total_ctx_tokens).view(1, -1, 1).to(device)
    all_tokens = torch.cat([context, dyn_sf], dim=1)
    dT = all_tokens.transpose(1, 2)  # (1, 1, L)
    attn_bias = torch.where(all_tokens >= dT, 0., -torch.inf).reshape(1, 1, all_tokens.shape[1], all_tokens.shape[1]).contiguous()
    lvl_emb = dyn_sf.view(1, L)

    return lvl_emb, attn_bias


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)


class VAR(nn.Module):
    def __init__(
        self,
        depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1, vocab_size=8192, Cvae=64,
        attn_l2_norm=True,
        c_patch_nums=None, d_patch_nums=None,
        total_length=16, context_length=1,
        flash_if_available=True, fused_if_available=True, 
        vq_model=None, device=None,
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0 
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads  # 16 1024 1024 16
        self.vq_model, self.V, self.Cvae, self.device = vq_model, vocab_size, Cvae, device
        
        if self.vq_model is not None:
            for p in self.vq_model.parameters():
                p.requires_grad = False
            self.vq_model.eval()
            
        self.total_length, self.context_length = total_length, context_length
        self.dynamic_length = self.total_length - self.context_length
        self.cond_drop_rate = cond_drop_rate # 0.1
        self.c_patch_nums, self.d_patch_nums = c_patch_nums, d_patch_nums
        self.num_stages_minus_1 = len(self.c_patch_nums) - 1
        self.rng = torch.Generator(device=self.device)
        squares = np.array(d_patch_nums) ** 2
        self.index_ranges = [(sum(squares[:i]), sum(squares[:i+1])) for i in range(len(squares))]

        # 1. patch embedding
        self.begin_ends, cur = [], 0
        for _ in range(self.dynamic_length):
            for _, pn in enumerate(self.d_patch_nums):
                self.begin_ends.append((cur, cur+pn ** 2))
                cur += pn ** 2
        self.first_l = self.d_patch_nums[0] ** 2
        self.L = cur
        self.word_embed = nn.Linear(self.Cvae, self.C)
        self.prefix_embed = nn.Linear(self.Cvae, self.C)

        # 2. start position embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.pos_start = nn.Parameter(torch.empty(1, self.dynamic_length, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)

        # 3. frame position embedding
        self.time_embed = nn.Parameter(torch.zeros(self.dynamic_length, self.C), requires_grad=False)
        
        # 4. absolute position embedding
        pos_1LC = []
        for _ in range(self.dynamic_length):
            for _, pn in enumerate(self.d_patch_nums):
                pe = torch.empty(1, pn*pn, self.C)
                nn.init.trunc_normal_(pe, mean=0, std=init_std)
                pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)

        # 5. level embedding
        self.lvl_embed = nn.Embedding(len(self.d_patch_nums * (self.dynamic_length)), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # 6. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate 
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        
        lvl_1L, attn_bias_for_masking = create_attn_bias_for_masking(patch_nums=self.d_patch_nums,  
                                                                     dynamic_length=self.dynamic_length,
                                                                     context_length=self.context_length,
                                                                     device=self.device,
                                                                     c_patch_nums=self.c_patch_nums)
        self.register_buffer('lvl_1L', lvl_1L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)

        # loss function
        self.train_loss = nn.CrossEntropyLoss(reduction='none')
        token_counts = [x**2 / len(self.d_patch_nums) for x in self.d_patch_nums]
        scaling_factor = len(self.d_patch_nums) / sum(token_counts)
        self.loss_weights = torch.tensor([x * scaling_factor * 0.3 for x in token_counts], device=self.device)
        
        # Context Positional Embedding
        ctx_dim = self.c_patch_nums[-1]**2
        self.pos_ctx_template = nn.Parameter(torch.empty(1, ctx_dim, self.C))
        nn.init.trunc_normal_(self.pos_ctx_template, mean=0, std=init_std)
        self.lvl_ctx_idx = len(self.d_patch_nums) - 1
    
    def _strip_frame_special_token(self, idx_BTL: torch.Tensor) -> torch.Tensor:
        """
        The `indices_d` of the Tokenizer for each frame is: [spc | dyn_tokens...]
        where the length of `dyn_tokens` = sum(pn^2 for pn in self.d_patch_nums)
        This function removes `spc` and ensures the length is aligned to `dyn_tokens_len`.
        """
        dyn_tokens_len = sum(pn * pn for pn in self.d_patch_nums)

        # idx_BTL: [B, T, L]
        if idx_BTL.dim() != 3:
            raise ValueError(f"idx_BTL must be [B,T,L], got {tuple(idx_BTL.shape)}")

        L = idx_BTL.shape[-1]

        if L == dyn_tokens_len + 1:
            return idx_BTL[:, :, 1:]  # drop spc

        if L == dyn_tokens_len:
            return idx_BTL

        if L > dyn_tokens_len:
            return idx_BTL[:, :, -dyn_tokens_len:]

        raise ValueError(f"Unexpected per-frame token length L={L}, expected {dyn_tokens_len} or {dyn_tokens_len+1}")

    def train(self, mode=True):
            super().train(mode)
            if self.vq_model is not None:
                self.vq_model.eval()
            return self
    
    def get_logits(self, h_or_h_and_residual, cond_BD=None):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        if cond_BD is not None:
            return self.head(self.head_nm(h.float(), cond_BD).float()).float()
        else:
            return self.head(self.head_nm(h.float()).float()).float()

    def idxBL_to_var_input(self, idx_BTL):
        idx_BTL = self._strip_frame_special_token(idx_BTL)
            
        assert idx_BTL.shape[1] == self.dynamic_length
        B, T, L = idx_BTL.shape
        
        expected = sum(pn * pn for pn in self.d_patch_nums)
        assert L == expected, f"idx_BTL per-frame L={L}, expected {expected}"
        
        C, H, W = self.Cvae, self.c_patch_nums[-1], self.c_patch_nums[-1]
        next_scales, total_scales = [], len(self.d_patch_nums)

        offset = 0
        if hasattr(self.vq_model, 'num_vq_embeddings'):
            offset = self.vq_model.num_vq_embeddings

        for t in range(self.dynamic_length):
            z_dyn = torch.zeros(B, C, H, W).to(self.device)
            idx_BL = idx_BTL[:, t, :]
            for si, pn in enumerate(self.d_patch_nums[:-1]):
                s, e = self.index_ranges[si]
                idx = idx_BL[:, s:e] 

                idx_local = idx - offset
                idx_local = idx_local.clamp_(0, self.vq_model.num_dyn_embeddings - 1)
                idx = idx_local

                idx = idx.reshape(B, pn, pn)
                
                h_BChw = F.interpolate(self.vq_model.dynamics_quantize.embedding(idx).permute(0,3,1,2), size=(H, W), mode='bicubic')
                z_dyn = z_dyn + self.vq_model.dynamics_quantize.quant_resi[si/(total_scales -1)](h_BChw)
                pn_next = self.d_patch_nums[si+1]
                next_scales.append(F.interpolate(z_dyn, size=(pn_next, pn_next), mode='bilinear').view(B, C, -1).transpose(1, 2))

        return torch.cat(next_scales, dim=1) 

    @torch.no_grad()
    def autoregressive_infer_cfg(self, prefix, dyn, B, g_seed, top_k=100, top_p=1.0):
        # rng
        if g_seed is None:
            rng = None
        else:
            self.rng.manual_seed(g_seed)
            rng = self.rng

        # ---------- global/dyn vocab bookkeeping ----------
        offset = int(getattr(self.vq_model, "num_vq_embeddings", 0))
        dynV = int(getattr(self.vq_model, "num_dyn_embeddings", 0))
        assert dynV > 0, "vq_model.num_dyn_embeddings must be > 0"
        assert self.V >= offset + dynV, f"VAR vocab_size (V={self.V}) must cover [0, offset+dynV) = {offset+dynV}"

        # ---------- prepare positional embeddings ----------
        sos = self.pos_start.expand(B, -1, -1, -1)                         # [B, Tdyn, first_l, C]
        time_emb = self.time_embed.unsqueeze(0).unsqueeze(2)               # [1, Tdyn, 1, C]
        sos = sos + time_emb

        # lvl_pos covers ALL dynamic tokens across ALL frames (no special token)
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC               # [1, L_total, C]
        frame_token_len = sum(pn * pn for pn in self.d_patch_nums)         # per-frame tokens (no spc)

        # ---------- prefix embedding + ctx position ----------
        prefix_emb = self.prefix_embed(prefix)                             # [B, Lctx, C]
        spatial_pos = self.pos_ctx_template.repeat(1, self.context_length, 1)
        level_pos = self.lvl_embed(torch.tensor([self.lvl_ctx_idx], device=self.device)).unsqueeze(1)
        level_pos = level_pos.expand(1, prefix_emb.shape[1], -1)
        ctx_pos = (spatial_pos + level_pos).expand(B, -1, -1)
        prefix_emb = prefix_emb + ctx_pos

        # ---------- enable kv cache & prefill with prefix ----------
        for b in self.blocks:
            b.attn.kv_caching(True)

        x_prefill = prefix_emb
        for b in self.blocks:
            x_prefill = b(x=x_prefill, attn_bias=None)

        # ---------- autoregressive generation (frame-by-frame, stage-by-stage) ----------
        future_frames = []

        for i in range(self.dynamic_length):
            # absolute positions for this frame (no spc)
            lvl_frame_pos = lvl_pos[:, frame_token_len * i: frame_token_len * (i + 1)]  # [1, frame_token_len, C]

            # first stage token map: SOS + first stage absolute pos
            next_token_map = sos[:, i] + lvl_frame_pos[:, :self.first_l]                # [B, first_l, C]

            cur_L = 0
            f_hat = sos.new_zeros(B, self.Cvae, self.c_patch_nums[-1], self.c_patch_nums[-1])

            for si, pn in enumerate(self.d_patch_nums):
                cur_L += pn * pn

                # transformer blocks (kv cache is ON, so this appends tokens)
                x = next_token_map
                for b in self.blocks:
                    x = b(x=x, attn_bias=None)

                # logits over GLOBAL vocab
                logits_BlV = self.get_logits(x)                                         # [B, pn², Vglobal]

                # -------- critical: sample ONLY from dyn segment --------
                logits_dyn = logits_BlV[..., offset:offset + dynV]                      # [B, pn², dynV]
                idx_local = sample_with_top_k_top_p_(logits_dyn, rng=rng,
                                                    top_k=top_k, top_p=top_p, num_samples=1)  # [B, pn²], 0..dynV-1

                # embed with LOCAL indices (directly valid for dynamics_quantize.embedding)
                h_BChw = self.vq_model.dynamics_quantize.embedding(idx_local) \
                    .transpose(1, 2).reshape(B, self.Cvae, pn, pn)

                # update multi-scale latent + build next scale token map
                f_hat, next_token_map = self.vq_model.dynamics_quantize.generate_next_scale(
                    si, len(self.d_patch_nums), f_hat, h_BChw, HW=self.c_patch_nums[-1]
                )

                # prepare tokens for next stage
                if si != (len(self.d_patch_nums) - 1):
                    pn_next = self.d_patch_nums[si + 1]
                    next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)      # [B, pn_next², Cvae]

                    # correct shape: [B, 1, C]
                    time_frame_pos = self.time_embed[i].view(1, 1, -1).expand(B, -1, -1)

                    next_token_map = self.word_embed(next_token_map) \
                                    + lvl_frame_pos[:, cur_L: cur_L + pn_next * pn_next] \
                                    + time_frame_pos

            future_frames.append(f_hat)

        # ---------- disable kv cache ----------
        for b in self.blocks:
            b.attn.kv_caching(False)

        # ---------- decode: context + predicted dynamics ----------
        prefix_src = prefix.reshape(B, f_hat.shape[2], f_hat.shape[3], -1).permute(0, 3, 1, 2)
        prefix_src = self.vq_model.post_quant_conv(prefix_src)
        context_dec, cond_features = self.vq_model.decoder(prefix_src, return_features=True)

        if self.context_length > 1:
            cond_features = [
                f.reshape(B, self.context_length, *f.shape[-3:]).unsqueeze(1)
                .repeat(1, self.dynamic_length, 1, 1, 1, 1).reshape(-1, self.context_length, *f.shape[-3:])
                for f in cond_features
            ]
        else:
            cond_features = [
                f.unsqueeze(1).repeat(1, self.dynamic_length, 1, 1, 1).reshape(-1, *f.shape[-3:])
                for f in cond_features
            ]

        pred_frame = torch.stack(future_frames, dim=1)                                   # [B, Tdyn, Cvae, H, W]
        quant_d2 = self.vq_model.post_quant_convdyn(pred_frame.reshape(-1, *pred_frame.shape[-3:]))
        dec = self.vq_model.cond_decoder(quant_d2, cond_features)
        dec_frames = dec.reshape(B, self.dynamic_length, -1, dec.shape[-1], dec.shape[-1])

        full_frame = torch.cat([context_dec.unsqueeze(1), dec_frames], dim=1)
        return full_frame.clamp(0.0, 1.0)
    
    def forward(self, prefix, idx_BTL):
        """
        :param prefix: prefix  [B, L, c]
        :param idx_BTL: idxBL [B, T, L]
        :return: loss
        """
        idx_BTL = self._strip_frame_special_token(idx_BTL)

        if idx_BTL.shape[-1] > self.L:
            # print(f"Trimming idx_BTL from {idx_BTL.shape} to last {self.L} tokens.") # Debug print
            idx_BTL = idx_BTL[:, :, -self.L:] 
            
        B, ed = prefix.shape[0], self.L
        
        x_BLCv_wo_first_l = self.idxBL_to_var_input(idx_BTL) 
        
        x_frames = x_BLCv_wo_first_l.view(B, self.dynamic_length, -1, self.Cvae) 
        with torch.cuda.amp.autocast(enabled=False):
            sos = self.pos_start.expand(B, -1, -1, -1)
            x_frames_c = torch.cat((sos, self.word_embed(x_frames.float())), dim=2) 
            time_emb = self.time_embed.unsqueeze(0).unsqueeze(2) 
            x_frames_c = x_frames_c + time_emb 
            x_BLC = x_frames_c.view(B, -1, x_frames_c.shape[-1]) 
            x_BLC = x_BLC + self.lvl_embed(self.lvl_1L.expand(B, -1)) + self.pos_1LC.expand(B, -1, -1)

        attn_bias = self.attn_bias_for_masking 
        bg = attn_bias.shape[-1] - ed
        
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype 
        
        x_BLC = x_BLC.to(dtype=main_type)
        prefix = prefix.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)
        
        prefix_emb = self.prefix_embed(prefix)
        
        spatial_pos = self.pos_ctx_template.repeat(1, self.context_length, 1)
        level_pos = self.lvl_embed(torch.tensor([self.lvl_ctx_idx], device=self.device)).unsqueeze(1)
        level_pos = level_pos.expand(1, prefix_emb.shape[1], -1)
        
        ctx_pos = (spatial_pos + level_pos).expand(B, -1, -1)
        prefix_emb = prefix_emb + ctx_pos
        
        x_BLC = torch.cat((prefix_emb, x_BLC), dim=1)
        
        AdaLNSelfAttn.forward
        for _, b in enumerate(self.blocks):
            x_BLC = b(x=x_BLC, attn_bias=attn_bias)

        x_BLC_logits = self.get_logits(x_BLC[:, bg:, :].float())
        loss = dict()
        loss_backward = 0
        x_BTLC_logits = x_BLC_logits.view(B, self.dynamic_length, -1, self.V)
        
        for i, (start, end) in enumerate(self.index_ranges):
            x_BTLC_logits_i = x_BTLC_logits[:, :, start:end, :].clone().view(-1, self.V)
            
            idx_BTL_i = idx_BTL[:, :, start:end].clone().view(-1).clamp(0, self.V - 1)
            
            loss_i = self.train_loss(x_BTLC_logits_i, idx_BTL_i).view(B, -1).mean()
            loss[i] = loss_i * self.loss_weights[i]
            loss_backward += loss[i] 
       
        return loss_backward, loss

    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5     # init_std < 0: automated
        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        vq_modules = set()
        if self.vq_model is not None:
            vq_modules = set(self.vq_model.modules())

        for m in self.modules():
            if m in vq_modules: continue
            with_weight = hasattr(m, 'weight') and m.weight is not None  
            with_bias = hasattr(m, 'bias') and m.bias is not None        
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)       
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)       
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.) 
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)  
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)  
                if with_bias: m.bias.data.zero_()
        
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head) 
                self.head.bias.data.zero_() 
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head) 
                self.head[-1].bias.data.zero_() 
        
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln) 
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()

        depth = len(self.blocks)  
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))  
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))  
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias) 
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)  
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)  
                sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma) 
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_() 
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln) 
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)

        pos_embed = get_1d_sincos_pos_embed(self.time_embed.shape[1], self.time_embed.shape[0]) # [15, 1024]
        self.time_embed.data.copy_(torch.from_numpy(pos_embed).float())
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'


def get_1d_sincos_pos_embed(embed_dim, seq_len):
    pos = np.arange(seq_len, dtype=np.float32)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb