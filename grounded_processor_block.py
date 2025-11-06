import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Tuple, Dict, Any

# -----------------------------------------------------------------
# 确保你从 diffusers 或你的项目路径正确导入了这些
# -----------------------------------------------------------------
from diffusers.models.attention_processor import Attention
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen

# # 这三个函数是 Qwen-Image-Edit 的核心，你需要确保能导入它们
# # (下面是假设的导入路径，请替换为你项目中的实际路径)
# try:
#     from diffusers.models.qwen2.modeling_qwen2_vl import (
#         dispatch_attention_fn, 
#         apply_rotary_emb_qwen
#     )
# except ImportError:
#     print("Warning: Could not import Qwen functions. Using placeholders.")
#     # 定义占位符以便代码能被解析
#     def dispatch_attention_fn(*args, **kwargs):
#         return F.scaled_dot_product_attention(*args, **kwargs)
#     def apply_rotary_emb_qwen(x, *args, **kwargs):
#         return x
# # -----------------------------------------------------------------

class GroundedQwenAttnProcessor:
    """
    这是一个自定义的 AttnProcessor，它实现了你的“Grounded Attention”逻辑。
    它被设计为“热插拔”替换默认的 QwenDoubleStreamAttnProcessor2_0。
    
    它重写了 __call__ 方法，不依赖继承。
    """
    
    _attention_backend = None
    _parallel_config = None

    def __init__(self, switch_point_fraction: float = 0.5):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("GroundedQwenAttnProcessor requires PyTorch 2.0+")
        
        # 存储切换点 (例如 0.5 = 50%)
        self.switch_point_fraction = switch_point_fraction

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # 图像流 (latents + orig_img + ctrl_img)
        encoder_hidden_states: torch.FloatTensor = None,  # 文本流
        encoder_hidden_states_mask: torch.FloatTensor = None, # 文本填充掩码
        attention_mask: Optional[torch.FloatTensor] = None, # 外部传入的掩码
        image_rotary_emb: Optional[torch.Tensor] = None,
        **kwargs, # 通过 kwargs 捕获 joint_attention_kwargs
    ) -> torch.FloatTensor:
        
        if encoder_hidden_states is None:
            raise ValueError("GroundedQwenAttnProcessor requires encoder_hidden_states (text stream)")

        # --- 1. 从 kwargs 和 attn 模块中提取控制信号 ---
        latent_mask = kwargs.get("latent_mask", None) 
        seq_lengths = kwargs.get("seq_lengths", None)
        
        block_index = getattr(attn, "block_index", 0)
        total_blocks = getattr(attn, "total_blocks", 1) # 避免除零
        switch_block_index = int(total_blocks * self.switch_point_fraction)
        
        # --- 2. 复制 QKV 投影和 RoPE ---
        seq_txt = encoder_hidden_states.shape[1]

        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        # (省略 ... QKV unflatten, norm, RoPE ... )
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))
        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None: img_query = attn.norm_q(img_query)
        if attn.norm_k is not None: img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None: txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None: txt_key = attn.norm_added_k(txt_key)

        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)
            
        # --- 3. 内存高效的注意力控制 (OOM 解决方案) ---
        
        if latent_mask is not None and seq_lengths is not None:
            L_noise, L_orig, L_control = seq_lengths
            B, L_txt, H, D = txt_query.shape # 形状是 (B, L, H, D)
            
            Q_noise, Q_orig, Q_ctrl = torch.split(img_query, [L_noise, L_orig, L_control], dim=1)
            K_noise, K_orig, K_ctrl = torch.split(img_key, [L_noise, L_orig, L_control], dim=1)
            V_noise, V_orig, V_ctrl = torch.split(img_value, [L_noise, L_orig, L_control], dim=1)

            # "背景世界"：保持不变, 只看原图
            K_bg, V_bg = K_orig, V_orig # (B, L_orig, H, D)
            attn_mask_bg = None # K_bg 不包含 K_txt, 所以不需要文本掩码
            
            if block_index > switch_block_index:
                # --- v7: 早期 Blocks (逻辑修正) ---
                
                # "前景世界"：[K_txt, K_ctrl] (强制布局, 忽略原图)
                K_fg = torch.cat([txt_key, K_ctrl], dim=1)
                V_fg = torch.cat([txt_value, V_ctrl], dim=1)
                
                attn_mask_fg = None
                if encoder_hidden_states_mask is not None:
                    txt_mask_sdpa = encoder_hidden_states_mask.bool().unsqueeze(1).unsqueeze(1) # (B, 1, 1, L_txt)
                    L_non_txt = K_fg.shape[1] - L_txt # = L_control
                    attn_mask_fg = F.pad(txt_mask_sdpa, (0, L_non_txt), value=True) # (B, 1, 1, L_txt + L_ctrl)

                # Call 1 (BG Noise): Q_noise -> K_bg
                out_noise_bg = dispatch_attention_fn(
                    Q_noise, K_bg, V_bg,
                    attn_mask=attn_mask_bg,
                    backend=self._attention_backend, parallel_config=self._parallel_config
                )
                
                # Call 2 (FG Noise): Q_noise -> K_fg
                mask_noise_fg = attn_mask_fg.expand(-1, H, L_noise, -1) if attn_mask_fg is not None else None
                out_noise_fg = dispatch_attention_fn(
                    Q_noise, K_fg, V_fg,
                    attn_mask=mask_noise_fg,
                    backend=self._attention_backend, parallel_config=self._parallel_config
                )

                # Call 3 (Text + Ctrl): Q_txt_ctrl -> K_fg
                Q_txt_ctrl = torch.cat([txt_query, Q_ctrl], dim=1)
                mask_txt_ctrl = attn_mask_fg.expand(-1, H, Q_txt_ctrl.shape[1], -1) if attn_mask_fg is not None else None
                out_txt_ctrl = dispatch_attention_fn(
                    Q_txt_ctrl, K_fg, V_fg,
                    attn_mask=mask_txt_ctrl,
                    backend=self._attention_backend, parallel_config=self._parallel_config
                )

                # Call 4 (!!! 关键修复 !!!): Q_orig -> K_bg
                # 隔离 Q_orig，让它只关注原图(K_bg)，以保护风格信息
                out_orig = dispatch_attention_fn(
                    Q_orig, K_bg, V_bg,
                    attn_mask=attn_mask_bg,
                    backend=self._attention_backend, parallel_config=self._parallel_config
                )

                # --- 组合结果 (早期) ---
                fg_mask = latent_mask.bool().unsqueeze(-1).unsqueeze(-1) # (B, L_noise, 1, 1)
                out_noise = torch.where(
                    fg_mask, out_noise_fg, out_noise_bg
                )
                
                out_txt, out_ctrl = torch.split(out_txt_ctrl, [L_txt, L_control], dim=1)
                
                img_attn_output = torch.cat([out_noise, out_orig, out_ctrl], dim=1)
                txt_attn_output = out_txt
                
            else:
                # --- v7: 后续 Blocks (逻辑不变, v6 已正确) ---
                
                # "前景世界"：[K_txt, K_orig, K_ctrl] (引入风格)
                K_fg = torch.cat([txt_key, K_orig, K_ctrl], dim=1)
                V_fg = torch.cat([txt_value, V_orig, V_ctrl], dim=1)

                attn_mask_fg = None
                if encoder_hidden_states_mask is not None:
                    txt_mask_sdpa = encoder_hidden_states_mask.bool().unsqueeze(1).unsqueeze(1) # (B, 1, 1, L_txt)
                    L_non_txt = K_fg.shape[1] - L_txt # = L_orig + L_control
                    attn_mask_fg = F.pad(txt_mask_sdpa, (0, L_non_txt), value=True) # (B, 1, 1, L_k_fg)

                # Call 1 (BG Noise): Q_noise -> K_bg
                out_noise_bg = dispatch_attention_fn(
                    Q_noise, K_bg, V_bg,
                    attn_mask=attn_mask_bg, # None
                    backend=self._attention_backend, parallel_config=self._parallel_config
                )
                
                # Call 2 (FG Noise): Q_noise -> K_fg
                mask_noise_fg = attn_mask_fg.expand(-1, H, L_noise, -1) if attn_mask_fg is not None else None
                out_noise_fg = dispatch_attention_fn(
                    Q_noise, K_fg, V_fg,
                    attn_mask=mask_noise_fg,
                    backend=self._attention_backend, parallel_config=self._parallel_config
                )
                
                # Call 3 (Text + Orig + Ctrl): Q_others -> K_fg
                # Q 和 K 逻辑对齐
                Q_others = torch.cat([txt_query, Q_orig, Q_ctrl], dim=1)
                mask_others = attn_mask_fg.expand(-1, H, Q_others.shape[1], -1) if attn_mask_fg is not None else None
                out_others = dispatch_attention_fn(
                    Q_others, K_fg, V_fg,
                    attn_mask=mask_others,
                    backend=self._attention_backend, parallel_config=self._parallel_config
                )

                # --- 组合结果 (后续) ---
                fg_mask = latent_mask.bool().unsqueeze(-1).unsqueeze(-1) # (B, L_noise, 1, 1)
                out_noise = torch.where(
                    fg_mask, out_noise_fg, out_noise_bg
                )

                out_txt, out_orig, out_ctrl = torch.split(out_others, [L_txt, L_orig, L_control], dim=1)
                
                img_attn_output = torch.cat([out_noise, out_orig, out_ctrl], dim=1)
                txt_attn_output = out_txt

            # --- 结束 if/else block_index ---
            
            img_attn_output = img_attn_output.flatten(2, 3)
            txt_attn_output = txt_attn_output.flatten(2, 3)

        else:
            # --- 4. 修正后的回退逻辑 (内存高效) ---
            B, L_txt, H, D = txt_query.shape
            L_img = img_query.shape[1]
            
            K_joint = torch.cat([txt_key, img_key], dim=1) # (B, L_txt + L_img, H, D)
            V_joint = torch.cat([txt_value, img_value], dim=1) # (B, L_txt + L_img, H, D)

            attn_mask_joint = None
            if encoder_hidden_states_mask is not None:
                txt_mask = encoder_hidden_states_mask.bool().unsqueeze(1).unsqueeze(1) # (B, 1, 1, L_txt)
                L_k_total = K_joint.shape[1]
                L_k_img = L_k_total - L_txt
                kv_mask = F.pad(txt_mask, (0, L_k_img), value=True) # (B, 1, 1, L_k_total)
                
                L_q_total = L_txt + L_img
                attn_mask_joint = kv_mask.expand(-1, H, L_q_total, -1) # (B, H, L_q_total, L_k_total)

            # Call 1: "Text" Query -> K_joint
            Q_txt_split = txt_query
            mask_txt_q = attn_mask_joint[:, :, :L_txt, :] if attn_mask_joint is not None else None
            
            out_txt = dispatch_attention_fn(
                Q_txt_split, K_joint, V_joint,
                attn_mask=mask_txt_q,
                backend=self._attention_backend, parallel_config=self._parallel_config
            )

            # Call 2: "Image" Query -> K_joint
            Q_img_split = img_query
            mask_img_q = attn_mask_joint[:, :, L_txt:, :] if attn_mask_joint is not None else None
            
            out_img = dispatch_attention_fn(
                Q_img_split, K_joint, V_joint,
                attn_mask=mask_img_q,
                backend=self._attention_backend, parallel_config=self._parallel_config
            )
            
            # 组合结果
            txt_attn_output = out_txt.flatten(2, 3)
            img_attn_output = out_img.flatten(2, 3)


        # --- 5. 复制输出投影 ---
        img_attn_output = img_attn_output.to(img_query.dtype)
        txt_attn_output = txt_attn_output.to(txt_query.dtype)

        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)

        txt_attn_output = attn.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output