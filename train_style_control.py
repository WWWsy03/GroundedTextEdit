import argparse
import copy
import gc
import logging
import os
import math
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from PIL import Image
import numpy as np

from accelerate import Accelerator, init_empty_weights
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
import datasets
import diffusers
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
import transformers
from omegaconf import OmegaConf

from style_transfer_pipeline import QwenImageEditPlusPipelineWithStyleControl
from style_transfer_processor import QwenDoubleStreamAttnProcessor2_0WithStyleControl

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.loaders import QwenImageLoraLoaderMixin
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.models import AutoencoderKLQwenImage, QwenImageTransformer2DModel
from typing import Union, List, Optional, Dict, Any, Callable
import torch.nn as nn
from diffusers.utils.torch_utils import randn_tensor
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor
from style_transfer_pipeline import (
    retrieve_latents,
    calculate_dimensions,
    CONDITION_IMAGE_SIZE, # 假设这些常量在你导入时可用
    VAE_IMAGE_SIZE,       # 假设这些常量在你导入时可用
    calculate_shift,
    retrieve_timesteps,
)
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor
from style_transfer_processor import Attention
from diffusers.utils import is_torch_xla_available, replace_example_docstring
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor

import logging
logger = get_logger(__name__, log_level="INFO")


# ----------------------------------------
# 辅助函数 (来自参考脚本)
# ----------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train Qwen-Image-Edit with Style Control.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="Path to the training config file (OmegaConf YAML).",
    )
    args = parser.parse_args()
    return args.config

# 你的 pipeline 和参考脚本都依赖这个
def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    
    # Qwen-Image VAE 需要 16*2 = 32 的倍数
    multiple_of = 32 
    width = round(width / multiple_of) * multiple_of
    height = round(height / multiple_of) * multiple_of

    return width, height

# ----------------------------------------
# 数据集定义
# ----------------------------------------

class StyleEditDataset(Dataset):
    def __init__(self, train_data_dir, content_folder="content_images", style_folder="style_images", gt_folder="ground_truth_images", prompt_file="prompts.txt"):
        self.data_dir = Path(train_data_dir)
        self.content_dir = self.data_dir / content_folder
        self.style_dir = self.data_dir / style_folder
        self.gt_dir = self.data_dir / gt_folder
        prompt_path = self.data_dir / prompt_file

        if not all([self.content_dir.exists(), self.style_dir.exists(), self.gt_dir.exists(), prompt_path.exists()]):
            raise FileNotFoundError(f"Dataset directories not found in {train_data_dir}")

        # 1. 加载 Prompts
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.prompts = [line.strip() for line in f.readlines() if line.strip()]

        # 2. 匹配图像
        # 假设图像文件名一一对应 (e.g., 001.jpg, 002.jpg)
        self.image_files = []
        for i in range(len(self.prompts)):
            # 尝试查找匹配的文件，假设基于索引或共同的文件名
            # 为简单起见，我们假设第 i 行 prompt 对应第 i 个文件
            # 你需要调整这个逻辑以匹配你的文件名 (e.g., img1.jpg, style1.jpg, gt1.jpg)
            
            # 示例：假设文件名是 1.jpg, 2.jpg...
            # filename = f"{i+1}.jpg" 
            
            # 示例：假设文件名与 prompt 列表顺序一致
            # 我们需要一个排序过的文件列表
            content_files = sorted([f for f in self.content_dir.glob('*.jpg')] + [f for f in self.content_dir.glob('*.png')])
            style_files = sorted([f for f in self.style_dir.glob('*.jpg')] + [f for f in self.style_dir.glob('*.png')])
            gt_files = sorted([f for f in self.gt_dir.glob('*.jpg')] + [f for f in self.gt_dir.glob('*.png')])

            if i < len(content_files) and i < len(style_files) and i < len(gt_files):
                self.image_files.append({
                    "content": content_files[i],
                    "style": style_files[i],
                    "gt": gt_files[i]
                })
            else:
                logger.warning(f"Skipping index {i} due to missing image files.")

        if len(self.prompts) != len(self.image_files):
             logger.warning(f"Mismatch: {len(self.prompts)} prompts vs {len(self.image_files)} image sets. Truncating.")
             min_len = min(len(self.prompts), len(self.image_files))
             self.prompts = self.prompts[:min_len]
             self.image_files = self.image_files[:min_len]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        try:
            prompt = self.prompts[idx]
            files = self.image_files[idx]
            
            content_img = Image.open(files["content"]).convert("RGB")
            style_img = Image.open(files["style"]).convert("RGB")
            gt_img = Image.open(files["gt"]).convert("RGB")
            print(f"Loaded data index {idx}: content {files['content'].name}, style {files['style'].name}, gt {files['gt'].name}")
            
            return content_img, style_img, gt_img, prompt
        except Exception as e:
            logger.error(f"Error loading data at index {idx}: {e}")
            # 尝试加载下一个
            return self.__getitem__((idx + 1) % len(self))

def collate_fn(examples):
    content_images = [e[0] for e in examples]
    style_images = [e[1] for e in examples]
    gt_images = [e[2] for e in examples]
    prompts = [e[3] for e in examples]
    print(f"Collate batch size: {len(prompts)}")
    
    return {
        "content_images_pil": content_images,
        "style_images_pil": style_images,
        "gt_images_pil": gt_images,
        "prompts": prompts
    }

# ----------------------------------------
# 主训练函数
# ----------------------------------------

def main():
    config_path = parse_args()
    args = OmegaConf.load(config_path)

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    # 1. 初始化 Accelerator (DeepSpeed 配置从 `accelerate config` 自动读取)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    # 1. 初始化 Accelerator (标准 DDP，不需要 DeepSpeed/FSDP)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir),
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%MS",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # 设置 DType
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    # --- 2. 加载模型 ---
    logger.info("Loading models...")
    
    tokenizer = Qwen2Tokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    processor = Qwen2VLProcessor.from_pretrained(args.pretrained_model_name_or_path, subfolder="processor")
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    
    # [关键 1] VAE 加载到 GPU (它比较小，且频繁使用)
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=weight_dtype
    ).to(accelerator.device)
    
    # [关键 2] Text Encoder 强制保留在 CPU
    # Qwen2.5-VL 很大，为了省显存，我们让它留在 CPU 上
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=weight_dtype
    ).to("cpu") 
    
    # Transformer 加载到 GPU (我们需要训练它)
    transformer = QwenImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=weight_dtype
    ).to(accelerator.device)

    # --- 3. 组装 Pipeline ---
    # 注意：pipeline 内部会持有这些模型。
    # 此时 pipeline.text_encoder 在 CPU，其他在 GPU。
    pipeline = QwenImageEditPlusPipelineWithStyleControl(
        scheduler=noise_scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        processor=processor,
        transformer=transformer,
    )
    # 只有 pipeline 自身不需要 .to(device)，因为它混合了设备

    # --- 4. 冻结 & 解冻 ---
    logger.info("🔒 正在冻结主干网络参数...")
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.transformer.requires_grad_(False)
    
    trainable_params = []
    if hasattr(pipeline.transformer, "transformer_blocks"):
        for i, block in enumerate(pipeline.transformer.transformer_blocks):
            processor = block.attn.processor
            if isinstance(processor, QwenDoubleStreamAttnProcessor2_0WithStyleControl):
                if not isinstance(processor, nn.Module):
                     raise TypeError(f"Processor must be nn.Module")
                for param_name, param in processor.named_parameters():
                    if "style_k_proj" in param_name or "style_v_proj" in param_name:
                        param.requires_grad = True
                        trainable_params.append(param)
    
    if not trainable_params:
        raise ValueError("No trainable parameters found!")
    logger.info(f"💰 可训练参数量: {sum(p.numel() for p in trainable_params) / 1_000_000:.2f} M")

    # --- 5. 优化器 ---
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # --- 6. 数据集 ---
    train_dataset = StyleEditDataset(train_data_dir=args.data_config.train_data_dir)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # --- 7. Accelerator Prepare (标准 DDP) ---
    # 我们只准备需要训练的部分。
    # VAE (GPU, Frozen) 和 TextEncoder (CPU, Frozen) 不需要 prepare
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        pipeline.transformer, optimizer, train_dataloader, lr_scheduler
    )
    
    # 回填 pipeline.transformer (它被 DDP 包装了)
    pipeline.transformer = transformer

    # --- 8. 训练准备 ---
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    
    # config 获取
    vae_config = pipeline.vae.config
    vae_scale_factor = 2 ** len(vae_config.temperal_downsample)
    latents_mean = torch.tensor(vae_config.latents_mean).view(1, 1, vae_config.z_dim, 1, 1).to(accelerator.device, dtype=weight_dtype)
    latents_std_inv = 1.0 / torch.tensor(vae_config.latents_std).view(1, 1, vae_config.z_dim, 1, 1).to(accelerator.device, dtype=weight_dtype)

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # 训练循环
    global_step = 0
    logger.info("***** Running training (CPU Offload Strategy) *****")
    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps", disable=not accelerator.is_local_main_process)

    # 确保模式正确
    pipeline.vae.eval()
    pipeline.text_encoder.eval()

    for epoch in range(args.num_train_epochs):
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            
            with torch.no_grad():
                bsz = len(batch["prompts"])
                device = accelerator.device # GPU
                
                # [1] Text Encoding (在 CPU 上进行)
                # 我们需要构建输入，并确保它们在 CPU 上
                # 你的 encode_prompt 可能会自动把输入移到 text_encoder 的设备 (CPU)
                # 或者我们需要手动处理。为了安全，我们手动处理 condition_images 的设备。
                
                content_images_pil = batch["content_images_pil"]
                style_images_pil = batch["style_images_pil"]
                condition_images = []
                for img_list in zip(content_images_pil, style_images_pil):
                    img_pair = []
                    for img in img_list:
                        w, h = calculate_dimensions(CONDITION_IMAGE_SIZE, img.size[0] / img.size[1])
                        # resize 返回 PIL image 或 tensor
                        processed_img = pipeline.image_processor.resize(img, h, w)
                        img_pair.append(processed_img)
                    condition_images.append(img_pair[0])
                    condition_images.append(img_pair[1])

                # 调用 encode_prompt。关键是传入 device="cpu" (如果 text_encoder 需要)
                # 或者让 pipeline 自动检测 text_encoder.device
                # 你的 _get_qwen_prompt_embeds 默认用 self.text_encoder.device
                # 但为了确保结果能回到 GPU，我们在 encode_prompt 返回后手动 .to(device)
                
                # 注意：encode_prompt 内部可能调用了 pipeline.processor (Qwen2VLProcessor)
                # 它通常处理 PIL 图片。如果传入 Tensor，需确保在 CPU。
                
                # 执行 CPU 计算
                prompt_embeds_cpu, prompt_embeds_mask_cpu = pipeline.encode_prompt(
                    image=condition_images, 
                    prompt=batch["prompts"], 
                    device="cpu", # 强制在 CPU 计算
                    num_images_per_prompt=1,
                    max_sequence_length=pipeline.tokenizer_max_length,
                )
                
                # [关键] 将计算结果移回 GPU
                prompt_embeds = prompt_embeds_cpu.to(device)
                prompt_embeds_mask = prompt_embeds_mask_cpu.to(device)
                txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()

                # [2] VAE Encoding (在 GPU 上进行，因为 VAE 较小且在 GPU 上)
                gt_images_pil = batch["gt_images_pil"]
                gt_pixel_values_list = []
                for img in gt_images_pil:
                    w, h = calculate_dimensions(VAE_IMAGE_SIZE, img.size[0] / img.size[1])
                    gt_pixel_values_list.append(
                        pipeline.image_processor.preprocess(img, h, w).unsqueeze(2)
                    )
                gt_pixel_values = torch.cat(gt_pixel_values_list, dim=0).to(device, dtype=weight_dtype)
                
                pixel_latents = pipeline.vae.encode(gt_pixel_values).latent_dist.sample()
                pixel_latents = (pixel_latents - latents_mean) * latents_std_inv

                # [修复 1] 处理 5D Latents (B, C, T, H, W) -> (B, C, H, W)
                # Qwen VAE 是 3D 的，如果是单张图片，T=1，我们需要把 T 维度挤掉
                if pixel_latents.ndim == 5:
                    pixel_latents = pixel_latents.squeeze(2) 
                
                # 现在它是 4D，shape[2] 是 H，shape[3] 是 W
                target_height = pixel_latents.shape[2]
                target_width = pixel_latents.shape[3]

                # --- B. Noise & Timesteps ---
                noise = torch.randn_like(pixel_latents, device=device, dtype=weight_dtype)
                u = compute_density_for_timestep_sampling(
                    weighting_scheme="none", batch_size=bsz, logit_mean=0.0, logit_std=1.0, mode_scale=1.29,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=device)
                sigmas = get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
                noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise
                
                # [关键修复 2] 传入 shape[1] 作为通道数
                packed_noisy_model_input = pipeline._pack_latents(
                    noisy_model_input, 
                    bsz, 
                    noisy_model_input.shape[1], # Channel
                    target_height, 
                    target_width,
                )
                L_noise = packed_noisy_model_input.shape[1]

                # --- C. Text Condition ---
                content_images_pil = batch["content_images_pil"]
                style_images_pil = batch["style_images_pil"]
                condition_images = []
                for img_list in zip(content_images_pil, style_images_pil):
                    img_pair = []
                    for img in img_list:
                        w, h = calculate_dimensions(CONDITION_IMAGE_SIZE, img.size[0] / img.size[1])
                        img_pair.append(pipeline.image_processor.resize(img, h, w))
                    condition_images.append(img_pair[0])
                    condition_images.append(img_pair[1])
                
                prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
                    image=condition_images, prompt=batch["prompts"], device=device, num_images_per_prompt=1,
                    max_sequence_length=pipeline.tokenizer_max_length,
                )
                txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()

                # --- D. Style Condition ---
                vae_images_list_content = []
                vae_images_list_style = []
                img_shapes_list = []

                for content_img, style_img in zip(content_images_pil, style_images_pil):
                    w_c, h_c = calculate_dimensions(VAE_IMAGE_SIZE, content_img.size[0] / content_img.size[1])
                    vae_images_list_content.append(pipeline.image_processor.preprocess(content_img, h_c, w_c).unsqueeze(2))
                    
                    w_s, h_s = calculate_dimensions(VAE_IMAGE_SIZE, style_img.size[0] / style_img.size[1])
                    vae_images_list_style.append(pipeline.image_processor.preprocess(style_img, h_s, w_s).unsqueeze(2))
                    
                    # RoPE 需要原始分辨率的比例，这里计算一下
                    noise_shape = (target_height // 2, target_width // 2)
                    # 注意：这里我们根据 latent 的缩放因子计算
                    content_shape = (h_c // (vae_scale_factor * 2) // 2, w_c // (vae_scale_factor * 2) // 2)
                    style_shape = (h_s // (vae_scale_factor * 2) // 2, w_s // (vae_scale_factor * 2) // 2)
                    img_shapes_list.append([(1, *noise_shape), (1, *content_shape), (1, *style_shape)])

                vae_images_content = torch.cat(vae_images_list_content, dim=0).to(device, dtype=weight_dtype)
                vae_images_style = torch.cat(vae_images_list_style, dim=0).to(device, dtype=weight_dtype)

                content_latents = pipeline._encode_vae_image(vae_images_content, generator=None)
                style_latents_unpacked = pipeline._encode_vae_image(vae_images_style, generator=None)
                
                # [关键修复 3] 同样处理 Style Latents 的 5D -> 4D
                if content_latents.ndim == 5:
                    content_latents = content_latents.squeeze(2)
                if style_latents_unpacked.ndim == 5:
                    style_latents_unpacked = style_latents_unpacked.squeeze(2)

                packed_content_latents = pipeline._pack_latents(
                    content_latents, bsz, content_latents.shape[1], content_latents.shape[2], content_latents.shape[3]
                )
                packed_style_latents = pipeline._pack_latents(
                    style_latents_unpacked, bsz, style_latents_unpacked.shape[1], style_latents_unpacked.shape[2], style_latents_unpacked.shape[3]
                )
                
                image_latents = torch.cat([packed_content_latents, packed_style_latents], dim=1)
                L_content_patches = packed_content_latents.shape[1]
                L_style_patches = packed_style_latents.shape[1]
                
                attention_kwargs = {
                    "style_image_latents": packed_style_latents,
                    "style_start_idx": L_noise + L_content_patches,
                    "style_end_idx": L_noise + L_content_patches + L_style_patches,
                    "noise_patches_length": L_noise,
                    "content_patches_length": L_content_patches,
                    "style_scale": args.style_scale
                }
                
                latent_model_input = torch.cat([packed_noisy_model_input, image_latents], dim=1)
                # 确保需要梯度 (针对某些特定的 DeepSpeed 配置)
                latent_model_input.requires_grad_(True)

            # --- 9.2 训练步骤 (Gradient) ---
            with accelerator.accumulate(pipeline.transformer):
                model_pred = pipeline.transformer(
                    hidden_states=latent_model_input,
                    timestep=timesteps / 1000,
                    guidance=None,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes_list,
                    txt_seq_lens=txt_seq_lens,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
                
                # Loss
                model_pred = model_pred[:, :L_noise]
                model_pred = pipeline._unpack_latents(
                    model_pred, 
                    height=target_height * vae_scale_factor, 
                    width=target_width * vae_scale_factor, 
                    vae_scale_factor=vae_scale_factor,
                )
                
                weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
                target = noise - pixel_latents
                
                # target 也是 4D (B, C, H, W)
                
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1), 1,
                )
                loss = loss.mean()
                
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # --- 9.3 日志和保存 ---
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss, "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        
                        unwrapped_transformer = accelerator.unwrap_model(pipeline.transformer)
                        style_control_state_dict = {}
                        for name, param in unwrapped_transformer.named_parameters():
                            if param.requires_grad:
                                style_control_state_dict[name] = param.cpu().to(torch.float32).detach()
                        
                        if style_control_state_dict:
                            torch.save(style_control_state_dict, os.path.join(save_path, "style_control_weights.pth"))
                            logger.info(f"✅ Saved weights to {save_path}")

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "final_model")
        os.makedirs(save_path, exist_ok=True)
        unwrapped_transformer = accelerator.unwrap_model(pipeline.transformer)
        style_control_state_dict = {}
        for name, param in unwrapped_transformer.named_parameters():
            if param.requires_grad:
                style_control_state_dict[name] = param.cpu().to(torch.float32).detach()
        if style_control_state_dict:
            torch.save(style_control_state_dict, os.path.join(save_path, "style_control_weights.pth"))
            logger.info("✅ Saved final weights.")

    accelerator.end_training()

if __name__ == "__main__":
    main()