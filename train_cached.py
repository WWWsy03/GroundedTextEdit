import argparse
import gc
import logging
import os
import math
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor
# 引入你的 Pipeline 和 Processor 类 (假设它们定义在 model_utils.py 或者直接粘贴在同一个文件)
# 为了演示，这里假设用户已经定义了这两个类，或者直接粘贴在代码最上方
# from model_utils import QwenImageEditPlusPipelineWithStyleControl, QwenDoubleStreamAttnProcessor2_0WithStyleControl
# *** 请确保将你提供的 Pipeline 和 Processor 类代码包含在运行环境中 ***

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.models import AutoencoderKLQwenImage, QwenImageTransformer2DModel
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor
from style_transfer_pipeline import QwenImageEditPlusPipelineWithStyleControl
from style_transfer_processor import QwenDoubleStreamAttnProcessor2_0WithStyleControl

logger = get_logger(__name__)

# ==============================================================================
# 2. 辅助函数
# ==============================================================================

def calculate_dimensions(target_area, ratio):
    # 复用 pipeline 逻辑中会用到的尺寸计算
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return width, height

# ==============================================================================
# 3. 数据集定义
# ==============================================================================

class StyleControlDataset(Dataset):
    def __init__(self, data_root, embeds_dir):
        self.data_root = Path(data_root)
        self.prompts_file = self.data_root / "prompts.txt"
        self.embeds_dir = Path(embeds_dir)
        
        # 简单的文件读取逻辑，请根据实际情况调整排序或匹配逻辑
        with open(self.prompts_file, "r", encoding="utf-8") as f:
            self.prompts = [line.strip() for line in f.readlines()]

        self.content_files = sorted(list((self.data_root / "content_images").glob("*")))
        self.style_files = sorted(list((self.data_root / "style_images").glob("*")))
        self.gt_files = sorted(list((self.data_root / "ground_truth_images").glob("*")))

        assert len(self.content_files) == len(self.prompts), "Content images and prompts mismatch"

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        # 直接返回 PIL Image，让 Pipeline 的 image_processor 去处理 Resize 和 Norm
        # 这样最安全，完全遵循 Pipeline 的预处理逻辑
        content_img = Image.open(self.content_files[idx]).convert("RGB")
        style_img = Image.open(self.style_files[idx]).convert("RGB")
        gt_img = Image.open(self.gt_files[idx]).convert("RGB")

        # 加载预计算的 Embeddings
        embed_path = self.embeds_dir / f"{idx}.pt"
        saved_embeds = torch.load(embed_path)

        return {
            "content_pil": content_img,
            "style_pil": style_img,
            "gt_pil": gt_img,
            "prompt_embeds": saved_embeds["prompt_embeds"],
            "prompt_embeds_mask": saved_embeds["prompt_embeds_mask"],
        }

    def collate_fn(self, examples):
        # 简单的 collate，把 PIL image 组成 list，Tensor stack 起来
        batch = {
            "content_pil": [example["content_pil"] for example in examples],
            "style_pil": [example["style_pil"] for example in examples],
            "gt_pil": [example["gt_pil"] for example in examples],
            "prompt_embeds": torch.stack([example["prompt_embeds"] for example in examples]),
            "prompt_embeds_mask": torch.stack([example["prompt_embeds_mask"] for example in examples]),
        }
        return batch

# ==============================================================================
# 4. 预计算逻辑 (Pre-computation)
# ==============================================================================

def precompute_embeddings(args, accelerator):
    """
    使用 '残血版' Pipeline (仅加载 Text Encoder) 计算 Prompt Embeddings 并缓存。
    完全复用 Pipeline 的 encode_prompt 逻辑。
    """
    if os.path.exists(args.precomputed_dir) and len(os.listdir(args.precomputed_dir)) > 0:
        logger.info("Found precomputed embeddings. Skipping...")
        return

    logger.info("Starting precomputation...")
    os.makedirs(args.precomputed_dir, exist_ok=True)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16": weight_dtype = torch.float16
    elif args.mixed_precision == "bf16": weight_dtype = torch.bfloat16

    # 1. 加载文本相关模型
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=weight_dtype
    )
    tokenizer = Qwen2Tokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    processor = Qwen2VLProcessor.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="processor"
    )

    # 2. 实例化 Pipeline (Transformer 和 VAE 传 None，节省显存)
    #    这样我们可以调用 encode_prompt, check_inputs, image_processor 等逻辑
    pipeline = QwenImageEditPlusPipelineWithStyleControl(
        scheduler=None, # 不不需要
        vae=None,       # 不需要
        transformer=None, # 不需要
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        processor=processor
    )
    pipeline.to(accelerator.device)

    # 3. 准备数据遍历
    prompts_file = os.path.join(args.train_data_dir, "prompts.txt")
    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f.readlines()]
    
    content_dir = os.path.join(args.train_data_dir, "content_images")
    content_files = sorted(list(Path(content_dir).glob("*")))

    # 4. 循环计算
    for idx, (prompt, img_path) in enumerate(tqdm(zip(prompts, content_files), total=len(prompts))):
        pil_img = Image.open(img_path).convert("RGB")
        
        # 重要：模拟 __call__ 中的预处理逻辑
        # Pipeline 的 encode_prompt 需要接收经过 resize 的图像，
        # 虽然 encode_prompt 内部也会处理，但为了和推理时 __call__ 的行为一致：
        # __call__ 中是先算 condition_width/height -> resize -> 传给 encode_prompt
        
        image_width, image_height = pil_img.size
        # 这里 1024*1024 应该做成参数
        CONDITION_IMAGE_SIZE = args.resolution * args.resolution 
        condition_width, condition_height = calculate_dimensions(
            CONDITION_IMAGE_SIZE, image_width / image_height
        )
        
        # 使用 pipeline 自带的 image_processor 进行 resize
        # 这确保了插值方法等细节一致
        condition_image = pipeline.image_processor.resize(pil_img, condition_height, condition_width)
        
        with torch.no_grad():
            # 调用 Pipeline 原生方法
            prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
                prompt=prompt,
                image=[condition_image], # encode_prompt 期望 list 或 tensor
                device=accelerator.device,
                num_images_per_prompt=1,
                max_sequence_length=args.max_sequence_length
            )
        
        # 保存
        torch.save({
            "prompt_embeds": prompt_embeds.cpu().squeeze(0), # remove batch dim [L, D]
            "prompt_embeds_mask": prompt_embeds_mask.cpu().squeeze(0)
        }, os.path.join(args.precomputed_dir, f"{idx}.pt"))

    # 5. 释放显存
    del pipeline
    del text_encoder
    del processor
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Precomputation finished & Memory cleared.")

# ==============================================================================
# 5. 主训练逻辑
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="/app/cold1/Qwen-Image-Edit-2509")
    parser.add_argument("--train_data_dir", type=str, default="/app/code/texteditRoPE/train_data_dir")
    parser.add_argument("--output_dir", type=str, default="/app/code/texteditRoPE/qwenimage-style-control-output")
    parser.add_argument("--precomputed_dir", type=str, default="/app/code/texteditRoPE/train_data_dir/cached_embeddings")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--max_sequence_length", type=int, default=1024)
    args = parser.parse_args()

    # 初始化 Accelerator
    project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=1,
        project_config=project_config
    )
    set_seed(0)

    # -------------------------------------------------------
    # Phase 1: 预计算 (Text Encoder)
    # -------------------------------------------------------
    if accelerator.is_main_process:
        precompute_embeddings(args, accelerator)
    accelerator.wait_for_everyone()
    print("✅ Pre-computation phase completed.")

    # -------------------------------------------------------
    # Phase 2: 准备训练模型
    # -------------------------------------------------------
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16": weight_dtype = torch.float16
    elif args.mixed_precision == "bf16": weight_dtype = torch.bfloat16

    # 加载 VAE, Scheduler, Transformer
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=weight_dtype
    )
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    transformer = QwenImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=weight_dtype
    )
    num_channels_latents = transformer.config['in_channels'] // 4

    # 冻结主干
    vae.requires_grad_(False)
    transformer.requires_grad_(False)

    # 实例化 "训练版" Pipeline
    # 目的：使用其内部 helper 方法 (_encode_vae_image, _pack_latents, prepare_latents 等)
    # Text Encoder 传 None 以节省显存
    pipeline = QwenImageEditPlusPipelineWithStyleControl(
        scheduler=noise_scheduler,
        vae=vae,
        transformer=transformer,
        text_encoder=None, # 已预计算
        tokenizer=None,
        processor=None
    )
    # 将 pipeline 移动到 device (主要是为了内部模块如 VAE 能在正确的 device 上)
    pipeline.to(accelerator.device)
    num_channels_latents = transformer.config['in_channels'] // 4 # **修正: 使用字典访问**
    # -------------------------------------------------------
    # Phase 3: 热插拔 Processor
    # -------------------------------------------------------
    if hasattr(transformer.config, 'hidden_size'):
        style_hidden_dim = transformer.config.hidden_size
    else:
        style_hidden_dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
    
    style_context_dim = 16 * 4 # 64

    trainable_params = []
    print("🔥 Injecting Trainable Processors...")
    
    for block in transformer.transformer_blocks:
        # 实例化你的 Processor
        processor = QwenDoubleStreamAttnProcessor2_0WithStyleControl(
            style_context_dim=style_context_dim,
            style_hidden_dim=style_hidden_dim
        )
        block.attn.processor = processor
        
        # 开启 KV 训练
        processor.style_k_proj.requires_grad_(True)
        processor.style_v_proj.requires_grad_(True)
        
        trainable_params.extend(processor.style_k_proj.parameters())
        trainable_params.extend(processor.style_v_proj.parameters())

    # -------------------------------------------------------
    # Phase 4: 优化器与数据
    # -------------------------------------------------------
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    
    dataset = StyleControlDataset(args.train_data_dir, embeds_dir=args.precomputed_dir)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.train_batch_size, 
        shuffle=True, 
        collate_fn=dataset.collate_fn,
        num_workers=4
    )

    # Accelerator Prepare
    # 注意：Pipeline 不是 torch.nn.Module 的标准子类，不能被 prepare。
    # 我们 prepare transformer 和 optimizer。
    transformer, optimizer, dataloader = accelerator.prepare(
        transformer, optimizer, dataloader
    )

    # -------------------------------------------------------
    # Phase 5: 训练循环
    # -------------------------------------------------------
    global_step = 0
    transformer.train()
    
    # 获取 VAE scale factor (复用 pipeline 逻辑)
    vae_scale_factor = pipeline.vae_scale_factor
    VAE_IMAGE_SIZE = args.resolution * args.resolution

    # 辅助函数：获取 Sigmas
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    print(f"Start Training. Params: {sum(p.numel() for p in trainable_params)/1e6:.2f}M")

    for epoch in range(10000):
        for batch in dataloader:
            with accelerator.accumulate(transformer):
                
                # 1. 数据准备 (PIL -> Resize/Process -> Tensor)
                content_imgs = batch["content_pil"]
                style_imgs = batch["style_pil"]
                gt_imgs = batch["gt_pil"]
                
                processed_vae_images_content = []
                processed_vae_images_style = []
                processed_vae_images_gt = []
                
                bsz = len(content_imgs)
                
                VAE_IMAGE_SIZE = args.resolution * args.resolution 

                for i in range(bsz):
                    w, h = content_imgs[i].size
                    vae_w, vae_h = calculate_dimensions(VAE_IMAGE_SIZE, w/h)
                    
                    # VaeImageProcessor.preprocess 输出 [C, H, W]
                    # 插入 T=1 维度，使用 unsqueeze(1) 得到 [C, T=1, H, W]
                    p_c = pipeline.image_processor.preprocess(content_imgs[i], vae_h, vae_w).unsqueeze(1)
                    p_s = pipeline.image_processor.preprocess(style_imgs[i], vae_h, vae_w).unsqueeze(1)
                    p_g = pipeline.image_processor.preprocess(gt_imgs[i], vae_h, vae_w).unsqueeze(1)
                    
                    processed_vae_images_content.append(p_c)
                    processed_vae_images_style.append(p_s)
                    processed_vae_images_gt.append(p_g)
                    
                    # 调试：检查单张图片张量是否是 [C, T, H, W] (例如 [3, 1, 1024, 1024])
                    # print(f"Single image tensor shape: {p_c.shape}") 

                # 最终堆叠成 Batch Tensor: [B, C, T, H, W]
                # **使用 torch.stack 自动在第 0 维添加 Batch size**
                vae_input_content = torch.stack(processed_vae_images_content).to(accelerator.device, dtype=weight_dtype)
                vae_input_style = torch.stack(processed_vae_images_style).to(accelerator.device, dtype=weight_dtype)
                vae_input_gt = torch.stack(processed_vae_images_gt).to(accelerator.device, dtype=weight_dtype)
                
                # 调试：检查最终 Batch 形状是否是 [B, C, T, H, W] (例如 [1, 3, 1, 1024, 1024])
                print(f"Final VAE input shape (Content): {vae_input_content.shape}")

                # 如果这里形状仍然是 6D，说明在某个隐藏的角落产生了多余的维度。
                # 最强力修正：如果形状 > 5D，强制移除多余的 Batch 维度 B2 (通常是第 1 维)
                if vae_input_content.ndim > 5:
                    # 我们假设多出来的 B2 维度是 1，我们需要将其移除 (squeeze)
                    vae_input_content = vae_input_content.squeeze(1)
                    vae_input_style = vae_input_style.squeeze(1)
                    vae_input_gt = vae_input_gt.squeeze(1)
                    print(f"Corrected VAE input shape (Content): {vae_input_content.shape}")
                # 调试语句 (可移除)
                print(f"Final VAE input shape: {vae_input_content.shape}")
                prompt_embeds = batch["prompt_embeds"].to(dtype=weight_dtype)
                prompt_embeds_mask = batch["prompt_embeds_mask"]

                # 2. 准备 Latents (利用 pipeline.prepare_latents 的逻辑)
                # 你的 prepare_latents 负责 VAE encode 和 Packing，并返回 indices
                # 但它内部假设要生成随机 latents。我们需要这里传入 GT latents 作为 base。
                # 所以我们只能借用 _encode_vae_image 和 _pack_latents，自己组装流程，
                # 否则直接调 prepare_latents 比较难塞入 GT 图像作为 "Noise" 的基础。
                
                bsz = len(content_imgs)
                print(f"transformer config: {transformer.config}")
                # num_channels_latents = transformer.config['in_channels'] // 4

                with torch.no_grad():
                    # Encode Content & Style (Condition)
                    # _encode_vae_image 处理了 retrieve_latents 和 norm
                    content_latents = pipeline._encode_vae_image(vae_input_content, generator=None)
                    style_latents = pipeline._encode_vae_image(vae_input_style, generator=None)
                    gt_latents_raw = pipeline._encode_vae_image(vae_input_gt, generator=None)
                    
                    # Pack Latents
                    # 需要获取 latent 的 H, W。 _encode_vae_image 返回 [B, C, 1, H, W]
                    l_h, l_w = gt_latents_raw.shape[3], gt_latents_raw.shape[4]

                    # Packing Logic (Directly calling static method or instance method)
                    packed_content = pipeline._pack_latents(content_latents, bsz, num_channels_latents, l_h, l_w)
                    packed_style = pipeline._pack_latents(style_latents, bsz, num_channels_latents, l_h, l_w)
                    packed_gt = pipeline._pack_latents(gt_latents_raw, bsz, num_channels_latents, l_h, l_w)
                    
                    L_content = packed_content.shape[1]
                    L_style = packed_style.shape[1]
                    L_noise = packed_gt.shape[1]

                # 3. Add Noise (Training Specific)
                noise = torch.randn_like(packed_gt) # 在 Packed 空间加噪，或者在 Latent 空间加噪再 Pack 是一样的
                # 为了严谨，我们在 Packed 空间做 Flow Matching
                
                u = compute_density_for_timestep_sampling(weighting_scheme="none", batch_size=bsz, logit_mean=0.0, logit_std=1.0, mode_scale=1.29)
                indices = (u * noise_scheduler.config.num_train_timesteps).long()
                timesteps = noise_scheduler.timesteps[indices].to(accelerator.device)
                
                sigmas = get_sigmas(timesteps, n_dim=packed_gt.ndim, dtype=packed_gt.dtype)
                packed_noisy_input = (1.0 - sigmas) * packed_gt + sigmas * noise
                
                # 4. 构造 Input & Attention Kwargs (完全对齐 Pipeline)
                # Pipeline Logic: latent_model_input = cat([latents(noise), content, style])
                hidden_states = torch.cat([packed_noisy_input, packed_content, packed_style], dim=1)
                
                # Indices logic
                style_start_idx = L_noise + L_content
                style_end_idx = style_start_idx + L_style

                attention_kwargs = {
                    "style_image_latents": packed_style,
                    "style_start_idx": style_start_idx,
                    "style_end_idx": style_end_idx,
                    "noise_patches_length": L_noise,
                    "content_patches_length": L_content,
                    "style_scale": 1.0
                }

                # 5. 构造 RoPE img_shapes
                # Pipeline: img_shapes = [[(1, h, w), (1, vh, vw)...]]
                # 对于训练，hidden_states 包含 Noise(GT size), Content, Style
                # 假设这三者在 VAE 编码后尺寸一致 (都经过 resize 到 resolution)
                # packed 后尺寸要除以 2
                p_h, p_w = l_h // 2, l_w // 2
                # 对应 [Noise, Content, Style]
                img_shapes = [[(1, p_h, p_w), (1, p_h, p_w), (1, p_h, p_w)]] * bsz
                
                txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()

                # 6. Forward
                if args.checkpointing_steps:
                    transformer.enable_gradient_checkpointing()
                
                model_pred = transformer(
                    hidden_states=hidden_states,
                    timestep=timesteps / 1000,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                # 7. Loss
                # 提取 Noise 部分的输出
                model_pred_noise = model_pred[:, :L_noise]
                target = noise - packed_gt

                weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
                loss = torch.mean(
                    (weighting.float() * (model_pred_noise.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                print(f"Step {global_step}: Loss {loss.item()}")

                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    # 只保存包含 style_ 的参数
                    unwrapped = accelerator.unwrap_model(transformer)
                    state_dict = unwrapped.state_dict()
                    style_weights = {k: v for k, v in state_dict.items() if "style_" in k}
                    torch.save(style_weights, os.path.join(save_path, "style_kv_weights.pt"))
                    logger.info(f"Saved style weights to {save_path}")

            if global_step >= args.max_train_steps:
                break
    
    accelerator.end_training()

if __name__ == "__main__":
    main()