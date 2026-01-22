from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from diffusers.models.autoencoders.vae import VectorQuantizer, VectorQuantizer2
from diffusers.configuration_utils import register_to_config, ConfigMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.accelerate_utils import apply_forward_hook

from .vae import Encoder, Decoder
from .conditional_vae import ConditionalEncoder, ConditionalDecoder


@dataclass
class CompressiveVQEncoderOutput(BaseOutput):

    latents: torch.FloatTensor
    dynamics_latents: torch.FloatTensor


@dataclass
class CompressiveVQDecoderOutput(BaseOutput):

    sample: torch.FloatTensor
    ref_sample: Optional[torch.FloatTensor] = None
    commit_loss: Optional[torch.FloatTensor] = None
    dyn_commit_loss: Optional[torch.FloatTensor] = None


class CompressiveVQModel(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 3,
        sample_size: int = 32,
        num_vq_embeddings: int = 256,
        norm_num_groups: int = 32,
        vq_embed_dim: Optional[int] = None,
        scaling_factor: float = 0.18215,
        norm_type: str = "group",  # group, spatial
        mid_block_add_attention=True,
        lookup_from_codebook=False,
        force_upcast=False,
        num_dyn_embeddings: int = 256,
        context_length: int = 1,
        max_att_resolution=32,
        resolution=256,
        patch_size=4,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.dyna_latent_channels = latent_channels
        self.context_length = context_length
        self.num_vq_embeddings = num_vq_embeddings
        self.num_dyn_embeddings = num_dyn_embeddings
        self.patch_size = patch_size

        # encoders
        self.cond_encoder = ConditionalEncoder(
            in_channels=in_channels,
            out_channels=self.dyna_latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=False,
            mid_block_add_attention=True,
            max_att_resolution=max_att_resolution,
            init_resolution=resolution,
            context_length=context_length,
        )

        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=False,
            mid_block_add_attention=mid_block_add_attention,
        )

        # vector quantization
        vq_embed_dim = vq_embed_dim if vq_embed_dim is not None else latent_channels
        self.vq_embed_dim = vq_embed_dim

        self.quant_conv = nn.Conv2d(latent_channels, vq_embed_dim, 1)
        self.quantize = VectorQuantizer2(
            num_vq_embeddings,
            vq_embed_dim,
            beta=1.0,
            # beta=0.25,
            remap=None,
            sane_index_shape=False,
            legacy=False,
            quant_resi=0.5, 
            share_quant_resi=4,
            default_qresi_counts=0,
            v_patch_nums=[1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
        )
        self.post_quant_conv = nn.Conv2d(vq_embed_dim, latent_channels, 1)

        self.quant_convdyn = nn.Conv2d(self.dyna_latent_channels, vq_embed_dim, 1)
        self.dynamics_quantize = VectorQuantizer2(
            num_dyn_embeddings,
            vq_embed_dim,
            beta=1.0,
            # beta=0.25,
            remap=None,
            sane_index_shape=False,
            legacy=False,
            quant_resi=0.5, 
            share_quant_resi=4,
            default_qresi_counts=0,
            v_patch_nums=[1, 2, 3, 4, 5, 6] # [1, 2, 3, 4, 5, 6, 8, 10] for higher resolutions
        )

        self.post_quant_convdyn = nn.Conv2d(vq_embed_dim, self.dyna_latent_channels, 1)

        # decoders
        self.cond_decoder = ConditionalDecoder(
            in_channels=self.dyna_latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_type=norm_type,
            mid_block_add_attention=True,
            max_att_resolution=max_att_resolution,
            init_resolution=16,  # TODO: magic number
            context_length=context_length,
        )

        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_type=norm_type,
            mid_block_add_attention=mid_block_add_attention,
        )

    def set_context_length(self, context_length):
        self.context_length = context_length
        self.config['context_length'] = context_length
        self.cond_encoder.set_context_length(context_length)
        self.cond_decoder.set_context_length(context_length)

    def init_modules(self):
        print(self.cond_decoder.load_state_dict(self.decoder.state_dict(), strict=False))
        print(self.cond_encoder.load_state_dict(self.encoder.state_dict(), strict=False))


    @apply_forward_hook
    def tokenize(self, pixel_values: torch.FloatTensor, context_length: int = 0):
        assert context_length == self.context_length
        B, T, C, H, W = pixel_values.shape
        context_frames = pixel_values[:, :context_length].reshape(-1, C, H, W)
        future_frames = pixel_values[:, context_length:].reshape(-1, C, H, W)
        future_length = T - context_length

        # 1. Encode Context
        h, cond_features = self.encoder(context_frames, return_features=True)
        # 处理 condition (与之前保持一致)
        if self.context_length > 1:
            B_fut = future_frames.shape[0] // future_length
            cond_features = [
                f.reshape(B_fut, self.context_length, *f.shape[-3:]).unsqueeze(1)
                .repeat(1, future_length, 1, 1, 1, 1).reshape(-1, self.context_length, *f.shape[-3:])
                for f in cond_features
            ]
        else:
            cond_features = [
                f.unsqueeze(1).repeat(1, future_length, 1, 1, 1).reshape(-1, *f.shape[-3:])
                for f in cond_features
            ]
            
        h = self.quant_conv(h)
        quant, commit_loss, info = self.quantize(h)

        # 2. Encode Dynamics
        d = self.cond_encoder(future_frames, cond_features)
        d = self.quant_convdyn(d)
        quant_d, dyn_commit_loss, info_d = self.dynamics_quantize(d)

        # 3. Construct Indices
        # info[0] 通常是 list of indices per scale
        indices_c = torch.cat([idx.reshape(B, context_length, -1) for idx in info[0]], dim=-1)
        indices_d = torch.cat([idx.reshape(B, future_length, -1) for idx in info_d[0]], dim=-1)
        
        # [FIX 1] 加上偏移量，让 dynamics token 在全局词表中独立
        indices_d = indices_d + self.num_vq_embeddings 

        prefix = info[1].permute(0, 2, 3, 1).reshape(B, -1, info[1].shape[1])
        dyn = info_d[1]

        # 4. Add Special Tokens
        spc_token = self.num_vq_embeddings - 1 
        spc_tokens = torch.ones(B, future_length, 1).to(indices_d.device, indices_d.dtype) * spc_token
        indices_d = torch.cat([spc_tokens, indices_d], dim=-1)

        return prefix, dyn, indices_c, indices_d

    @apply_forward_hook
    def detokenize(self, indices_c, indices_d, context_length, cache=None, return_cache=False):
        assert context_length == self.context_length
        B = indices_c.shape[0]
        
        # Flatten indices
        flat_indices_c = indices_c.reshape(B, -1)
        flat_indices_d = indices_d.reshape(B, -1)
        indices = torch.cat([flat_indices_c, flat_indices_d], dim=1)
        
        # 计算每个部分的 Token 长度
        ctx_tokens_per_frame = sum(p*p for p in self.quantize.v_patch_nums)
        dyn_tokens_per_frame = sum(p*p for p in self.dynamics_quantize.v_patch_nums)
        target_res_dyn = dyn_tokens_per_frame + 1 # +1 for special token

        start = 0
        
        # --- Decode Context (Fix: Handle Multiple Frames) ---
        context_dec_frames = []
        # 如果 indices_c 已经被 flatten 了，我们需要按帧切分
        # 注意：这里假设 context 部分没有 special token
        
        for t in range(context_length):
            # 取出当前帧的所有 scale token
            current_frame_indices = indices[:, start : start + ctx_tokens_per_frame]
            start += ctx_tokens_per_frame
            
            z_hat = torch.zeros(B, 64, 16, 16).to(indices.device)
            scale_indices = []
            frame_start = 0
            
            # 分割 scales
            for ph in self.quantize.v_patch_nums:
                num_tokens = ph * ph
                scale_indices.append(current_frame_indices[:, frame_start : frame_start + num_tokens])
                frame_start += num_tokens

            # 重建 Latent
            for si, ph in enumerate(self.quantize.v_patch_nums):
                idx = scale_indices[si].reshape(B, ph, ph)
                quant = self.quantize.embedding(idx).permute(0, 3, 1, 2)
                quant = F.interpolate(quant, size=(16, 16), mode='bicubic').contiguous()
                
                total_scales = len(self.quantize.v_patch_nums)
                # 注意：确保 quant_resi 支持 float index 或者这里逻辑与 VAR 一致
                scale_idx = si / (total_scales - 1) if total_scales > 1 else 0
                quant_resi = self.quantize.quant_resi[scale_idx](quant)
                z_hat += quant_resi
            
            quant2 = self.post_quant_conv(z_hat)
            # Decoder 输出
            dec_frame, _ = self.decoder(quant2, return_features=True)
            context_dec_frames.append(dec_frame)
            
            # 保存最后一帧的 cond_features 用于 Future (简化逻辑，通常取最后一帧或全部)
            if t == context_length - 1:
                _, cond_features = self.decoder(quant2, return_features=True)

        context_dec = torch.stack(context_dec_frames, dim=1) # [B, T_ctx, C, H, W]

        # --- Decode Future ---
        future_frames = []
        # 计算剩余长度包含了多少帧
        remaining_tokens = indices.shape[1] - start
        future_length = remaining_tokens // target_res_dyn
        
        for t in range(future_length):
            # 取出当前帧，跳过第1个 Special Token (+1)
            frame_start_idx = start + 1 
            frame_end_idx = start + target_res_dyn
            
            # [FIX 1] 对应 tokenize 中的加法，这里减去偏移量
            indices_d_t = (indices[:, frame_start_idx : frame_end_idx] - self.num_vq_embeddings).clamp(min=0, max=self.num_dyn_embeddings - 1)
            start += target_res_dyn # 移动总指针

            z_dyn = torch.zeros(B, self.dyna_latent_channels, 16, 16).to(indices.device)
            scale_indices = []
            scale_ptr = 0
            
            for ph in self.dynamics_quantize.v_patch_nums:
                num_tokens = ph * ph
                scale_indices.append(indices_d_t[:, scale_ptr : scale_ptr + num_tokens])
                scale_ptr += num_tokens

            for si, ph in enumerate(self.dynamics_quantize.v_patch_nums):
                idx = scale_indices[si].reshape(B, ph, ph)
                quant = self.dynamics_quantize.embedding(idx).permute(0, 3, 1, 2)
                
                if ph != 16: 
                    quant = F.interpolate(quant, size=(16, 16), mode='bicubic').contiguous()
                else:
                    quant = quant.contiguous()
                
                scale_factor = si / (len(self.dynamics_quantize.v_patch_nums) - 1) if len(self.dynamics_quantize.v_patch_nums) > 1 else 0
                quant_resi = self.dynamics_quantize.quant_resi[scale_factor](quant)
                z_dyn += quant_resi

            quant_d2 = self.post_quant_convdyn(z_dyn)
            
            # 注意：这里 cond_features 应该如何传递取决于 ConditionalDecoder 的实现
            # 如果是 Autoregressive，每一帧 condition 可能不同。如果是并行解码，使用 Context 特征即可。
            dec = self.cond_decoder(quant_d2, cond_features)
            future_frames.append(dec)

        if len(future_frames) > 0:
            dec = torch.stack(future_frames, dim=1)
            return torch.cat([context_dec, dec], dim=1)
        else:
            return context_dec
        
        
    @apply_forward_hook
    def encode(self, encoder, x: torch.FloatTensor, return_dict: bool = True) -> CompressiveVQEncoderOutput:
        h, d = encoder(x)
        h = self.quant_conv(h)

        if not return_dict:
            return (h, d)

        return CompressiveVQEncoderOutput(latents=h, dynamics_latents=d)

    @apply_forward_hook
    def decode(
        self, h: torch.FloatTensor, d: torch.FloatTensor,
        return_dict: bool = True,
    ) -> Union[CompressiveVQDecoderOutput, torch.FloatTensor]:
        # also go through quantization layer
        # quant, commit_loss, lvl_quant, _ = self.quantize(h)
        quant, commit_loss, _ = self.quantize(h)
        quant_d, dyn_commit_loss, _ = self.dynamics_quantize(d)

        quant2 = self.post_quant_conv(quant)
        quant2_d = self.post_quant_convdyn(quant_d)

        # ################# visual ！ #################
        # lvl_frame = []
        # for i in range(len(lvl_quant)):
        #     lvl_quant2 = self.post_quant_conv(lvl_quant[i])
        #     lvl_dec, _ = self.decoder(lvl_quant2, return_features=True)
        #     lvl_frame.append(lvl_dec)
        # lvl_frame_stack = torch.stack(lvl_frame, dim=0).squeeze(1) # [10 3 64 64]
        # import matplotlib.pyplot as plt
        # # 假设 pixel_values 是一个形状为 [1, 16, 3, 64, 64] 的张量
        # a = lvl_frame_stack  # 去掉批次维度，形状变为 [16, 3, 64, 64]
        # a = a.float()
        # row_image = torch.cat(list(a), dim=2)  # 沿宽度方向拼接
        # row_image_np = row_image.permute(1, 2, 0).cpu().detach().numpy()  # 转换为 [H, W, C]
        # plt.figure(figsize=(16, 4))  # 调整图像大小
        # plt.imshow(row_image_np)
        # plt.axis('off')  # 关闭坐标轴
        # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 调整子图布局，确保没有白边
        # plt.savefig('context_lvl.png', bbox_inches='tight', pad_inches=0)
        # ################# visual ！ #################

        ref_dec, cond_features = self.decoder(quant2, return_features=True)
        if self.context_length > 1:
            B = quant2_d.shape[0] // self.segment_len
            cond_features = [
                f.reshape(B, self.context_length, *f.shape[-3:]).unsqueeze(1).repeat(1, self.segment_len, 1, 1, 1, 1).reshape(-1, self.context_length, *f.shape[-3:]) for f in cond_features
            ]  # B*(T-t), t, C, H, W
        else:
            cond_features = [f.unsqueeze(1).repeat(1, self.segment_len, 1, 1, 1).reshape(-1,
                                                                                        *f.shape[-3:]) for f in cond_features]

        dec = self.cond_decoder(quant2_d, cond_features) # [7, 3, 64, 64]

        if not return_dict:
            return (
                dec,
                ref_dec,
                commit_loss,
                dyn_commit_loss,
            )

        return CompressiveVQDecoderOutput(sample=dec, ref_sample=ref_dec, commit_loss=commit_loss, dyn_commit_loss=dyn_commit_loss)

    def forward(
        self, sample: torch.FloatTensor, return_dict: bool = True, return_loss: bool = False,
        segment_len: int = None,
        dyn_sample: torch.FloatTensor = None,
    ) -> Union[CompressiveVQDecoderOutput, Tuple[torch.FloatTensor, ...]]:
        self.segment_len = segment_len

        # encode context frames
        h, cond_features = self.encoder(sample, return_features=True)
        if self.context_length > 1:
            B = dyn_sample.shape[0] // self.segment_len
            cond_features = [
                f.reshape(B, self.context_length, *f.shape[-3:]).unsqueeze(1).repeat(1, self.segment_len, 1, 1, 1, 1).reshape(-1, self.context_length, *f.shape[-3:]) for f in cond_features
            ]  # B*(T-t), t, C, H, W
        else:
            cond_features = [f.unsqueeze(1).repeat(1, self.segment_len, 1, 1, 1).reshape(-1,
                                                                                        *f.shape[-3:]) for f in cond_features]
        h = self.quant_conv(h)

        d = self.cond_encoder(dyn_sample, cond_features)
        d = self.quant_convdyn(d)

        dec = self.decode(h, d)

        if not return_dict:
            if return_loss:
                return (
                    dec.sample,
                    dec.ref_sample,
                    dec.commit_loss,
                    dec.dyn_commit_loss,
                )
            return (dec.sample,)
        if return_loss:
            return dec
        return CompressiveVQDecoderOutput(sample=dec.sample)
