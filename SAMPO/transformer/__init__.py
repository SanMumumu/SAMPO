from .action_model import HeadModelWithAction

from .var import VAR

from typing import Tuple
import torch.nn as nn


def build_var(
        # shared args
        vq_model, device,
        # VQ-VAE args
        c_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        d_patch_nums=(1, 2, 3, 4, 5, 6), # (1, 2, 3, 4, 5, 6, 8, 10) for higher resolution
        total_length=16, context_length=16, 
        # VAR args
        depth=16, shared_aln=False, attn_l2_norm=True,
        flash_if_available=True, fused_if_available=True,
        init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1,    # init_std < 0: automated
    ):
    heads, width, dpr = depth, depth * 64, 0.1 * depth/24
    
    # disable built-in initialization for speed
    for clz in (nn.Linear, nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d):
        setattr(clz, 'reset_parameters', lambda self: None)
    
    # build models
    var_wo_ddp = VAR(
        depth=depth, embed_dim=width, num_heads=heads, drop_rate=0., attn_drop_rate=0., drop_path_rate=dpr,
        norm_eps=1e-6, shared_aln=shared_aln, cond_drop_rate=0.1, vocab_size=8192+8192,
        attn_l2_norm=attn_l2_norm,
        c_patch_nums=c_patch_nums, d_patch_nums=d_patch_nums,
        total_length=total_length, context_length=context_length,
        flash_if_available=flash_if_available, fused_if_available=fused_if_available, 
        vq_model=vq_model, device=device,
    ).to(device)
    var_wo_ddp.init_weights(init_adaln=init_adaln, init_adaln_gamma=init_adaln_gamma, init_head=init_head, init_std=init_std)
    
    return var_wo_ddp
