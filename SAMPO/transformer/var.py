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

def create_attn_bias_for_masking(patch_nums, dynamic_length, context_length, device):
    patch_nums_d = patch_nums * dynamic_length
    L = sum(pn ** 2 for pn in patch_nums_d)

    dyn_sf = torch.cat([torch.full((pn * pn,), i) for i, pn in enumerate(patch_nums_d)]).view(1, L, 1).to(device)
    context = - torch.ones(context_length * 256).view(1, 256, 1).to(device) # magical number: prefix.shape[1]
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
        self.vq_model, self.V, self.Cvae, self.device = vq_model, vocab_size, Cvae, device  # 8192
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

        # 5. level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.d_patch_nums * (self.dynamic_length)), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # 6. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate 
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
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
        # # logger.info
        # print(
        #     f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
        #     f'[VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
        #     f'[drop ratios] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
        #     end='\n\n', flush=True
        # )
        
        # attention mask used in training (for masking out the future), it won't be used in inference, since kv cache is enabled
        lvl_1L, attn_bias_for_masking = create_attn_bias_for_masking(patch_nums=self.d_patch_nums,  # [1 1275], [1, 1, 1275, 1275]
                                                                     dynamic_length=self.dynamic_length,
                                                                     context_length=self.context_length,
                                                                     device=self.device)
        self.register_buffer('lvl_1L', lvl_1L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer) # LayerNorm + SiLU + Linear(in_features=1024, out_features=2048, bias=True)
        self.head = nn.Linear(self.C, self.V) # Linear(in_features=1024, out_features=4096, bias=True)

        # loss function
        self.train_loss = nn.CrossEntropyLoss(reduction='none')
        token_counts = [x**2 / len(self.d_patch_nums) for x in self.d_patch_nums]
        scaling_factor = len(self.d_patch_nums) / sum(token_counts)
        self.loss_weights = torch.tensor([x * scaling_factor for x in token_counts], device=self.device)
    
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
        assert idx_BTL.shape[1] == self.dynamic_length
        B, T, L = idx_BTL.shape
        C, H, W = self.Cvae, self.c_patch_nums[-1], self.c_patch_nums[-1]
        next_scales, total_scales = [], len(self.d_patch_nums)

        for t in range(self.dynamic_length):
            z_dyn = torch.zeros(B, C, H, W).to(self.device)
            idx_BL = idx_BTL[:, t, :]
            for si, pn in enumerate(self.d_patch_nums[:-1]):
                s, e = self.index_ranges[si]
                idx = idx_BL[:, s:e].reshape(B, pn, pn)
                h_BChw = F.interpolate(self.vq_model.dynamics_quantize.embedding(idx).permute(0,3,1,2), size=(H, W), mode='bicubic')
                z_dyn = z_dyn + self.vq_model.dynamics_quantize.quant_resi[si/(total_scales -1)](h_BChw)
                pn_next = self.d_patch_nums[si+1]
                next_scales.append(F.interpolate(z_dyn, size=(pn_next, pn_next), mode='area').view(B, C, -1).transpose(1, 2))

        return torch.cat(next_scales, dim=1) # [8, 1260, 64]
         
    @torch.no_grad()
    def autoregressive_infer_cfg(self, prefix, dyn, B, g_seed, top_k=100, top_p=1.0):#900 0.95 # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size 8
        :param g_seed: random seed
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        
        sos = self.pos_start.expand(B, -1, -1, -1) # [B, 15, 1, 1024]
        time_emb = self.time_embed.unsqueeze(0).unsqueeze(2) # [1, 15, 1, 1024]
        sos = sos + time_emb # add time embedding for each frame
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC # [1, 1275, 1024] + [1, 1275, 1024]

        future_frames = []
        prefix_emb = self.prefix_embed(prefix) # [8, 256, 1024]

        for b in self.blocks: b.attn.kv_caching(True)
        token_maps = []
        for i in range(self.dynamic_length):
            lvl_frame_pos = lvl_pos[:, 85*i : 85*(i+1)] # [1, 85, 1024] magic number
            next_token_map = sos[:, i] + lvl_frame_pos[:, :self.first_l] # [B, 1, 1024]
            if i == 0:
                for b in self.blocks:
                    prefix_emb = b(x=prefix_emb, attn_bias=None)

            cur_L = 0
            f_hat = sos.new_zeros(B, self.Cvae, self.c_patch_nums[-1], self.c_patch_nums[-1]) # [8, 64, 16, 16]
            
            cur_tf = 0
            for si, pn in enumerate(self.d_patch_nums):
                cur_L += pn*pn
                x = next_token_map
                AdaLNSelfAttn.forward
                for b in self.blocks:
                    x = b(x=x, attn_bias=None)

                logits_BlV = self.get_logits(x)
                
                # idx_Bl = dyn[:, i, cur_tf: cur_tf+pn*pn] # # TF
                # idx_Bl = logits_BlV.argmax(dim=-1)
                idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)
                h_BChw = self.vq_model.dynamics_quantize.embedding(idx_Bl).transpose_(1, 2).reshape(B, self.Cvae, pn, pn) # [B, Cvae, pn, pn]
                f_hat, next_token_map = self.vq_model.dynamics_quantize.generate_next_scale(si, len(self.d_patch_nums), f_hat, h_BChw, HW=self.c_patch_nums[-1])
                token_maps.append(f_hat)
                if si != (len(self.d_patch_nums)-1): # prepare for next stage
                    next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                    time_frame_pos = self.time_embed[i].unsqueeze(0).expand(B, -1, -1) # [B, 1, 1024]
                    next_token_map = self.word_embed(next_token_map) + lvl_frame_pos[:, cur_L:cur_L + self.d_patch_nums[si+1] ** 2] + time_frame_pos
                cur_tf += pn*pn # TF
            future_frames.append(f_hat)

        for b in self.blocks: b.attn.kv_caching(False)

        prefix_src = prefix.reshape(B, f_hat.shape[2], f_hat.shape[3], -1).permute(0, 3, 1, 2)
        prefix_src = self.vq_model.post_quant_conv(prefix_src)
        context_dec, cond_features = self.vq_model.decoder(prefix_src, return_features=True)
        if self.context_length > 1:
            cond_features = [
                f.reshape(B, self.context_length, *f.shape[-3:]).unsqueeze(1)
                .repeat(1, self.dynamic_length, 1, 1, 1, 1).reshape(-1, self.context_length, *f.shape[-3:])
                for f in cond_features
            ]  # B*(T-t), t, C, H, W
        else:
            cond_features = [f.unsqueeze(1).repeat(1, self.dynamic_length, 1, 1, 1)
                             .reshape(-1, *f.shape[-3:]) for f in cond_features]
            
        pred_frame = torch.stack(future_frames, dim=1) # [8, 15, 64, 16, 16]
        quant_d2 = self.vq_model.post_quant_convdyn(pred_frame.reshape(-1, *pred_frame.shape[-3:]))
        # quant_d2 = self.vq_model.post_quant_convdyn(dyn.float())
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
        B, ed = prefix.shape[0], self.L
        x_BLCv_wo_first_l = self.idxBL_to_var_input(idx_BTL) # [B, 15, 85]->[B, 1260, 64]
        
        x_frames = x_BLCv_wo_first_l.view(B, self.dynamic_length, -1, self.Cvae) # [B, 15, 84, 64]
        with torch.cuda.amp.autocast(enabled=False):
            sos = self.pos_start.expand(B, -1, -1, -1)
            x_frames_c = torch.cat((sos, self.word_embed(x_frames.float())), dim=2)                    # start token embedding
            time_emb = self.time_embed.unsqueeze(0).unsqueeze(2) # [1, 15, 1, 1024]
            x_frames_c = x_frames_c + time_emb                   # [B, 15, 85, 1024]                   # frame embedding
            x_BLC = x_frames_c.view(B, -1, x_frames_c.shape[-1]) # [B, 1275, 1024]
            x_BLC = x_BLC + self.lvl_embed(self.lvl_1L.expand(B, -1)) + self.pos_1LC.expand(B, -1, -1) # level embedding and position embedding(eacj)

        attn_bias = self.attn_bias_for_masking # attention map
        bg = attn_bias.shape[-1] - ed
        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype # bfloat16
        
        x_BLC = x_BLC.to(dtype=main_type)
        prefix = prefix.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)
        prefix_emb = self.prefix_embed(prefix)
        x_BLC = torch.cat((prefix_emb, x_BLC), dim=1) # add prefix information
        
        AdaLNSelfAttn.forward
        for _, b in enumerate(self.blocks):
            x_BLC = b(x=x_BLC, attn_bias=attn_bias)

        x_BLC_logits = self.get_logits(x_BLC[:, bg:, :].float())
        loss = dict()
        loss_backward = 0
        x_BTLC_logits = x_BLC_logits.view(B, self.dynamic_length, -1, self.V)
        for i, (start, end) in enumerate(self.index_ranges):
            x_BTLC_logits_i = x_BTLC_logits[:, :, start:end, :].clone().view(-1, self.V)
            idx_BTL_i = idx_BTL[:, :, start:end].clone().view(-1)
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

        pos_embed = get_1d_sincos_pos_embed(self.time_embed.shape[1], self.time_embed.shape[0])
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