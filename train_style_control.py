import argparse
import copy
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

from accelerate import Accelerator
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
from style_transfer_pipeline import (
    retrieve_latents,
    calculate_dimensions,
    CONDITION_IMAGE_SIZE, # 假设这些常量在你导入时可用
    VAE_IMAGE_SIZE,       # 假设这些常量在你导入时可用
    calculate_shift,
    retrieve_timesteps,
)
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

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # 日志记录
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # 设置 DType
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        print("使用 fp16 进行训练")
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        print("使用 bf16 进行训练")
        weight_dtype = torch.bfloat16

    # --- 1. 加载自定义 Pipeline ---
    # QwenImageEditPlusPipelineWithStyleControl 应该在 __init__ 中
    # 自动设置好自定义的 Processor
    try:
        pipeline = QwenImageEditPlusPipelineWithStyleControl.from_pretrained(
            args.pretrained_model_name_or_path,
            torch_dtype=weight_dtype,
            # local_files_only=True # 如果本地已经下载好
        )
    except Exception as e:
        logger.error(f"加载 pipeline 失败: {e}")
        logger.error("请确保你的 QwenImageEditPlusPipelineWithStyleControl 和 QwenDoubleStreamAttnProcessor2_0WithStyleControl 类已正确定义并导入。")
        raise

    # --- 2. 冻结主干网络，只训练 Processor 的新层 ---
    logger.info("🔒 正在冻结主干网络参数...")
    # 你必须分别冻结 pipeline 内的 nn.Module 组件
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.transformer.requires_grad_(False)

    trainable_params = []
    total_blocks = 0
    if hasattr(pipeline, "transformer") and hasattr(pipeline.transformer, "transformer_blocks"):
        total_blocks = len(pipeline.transformer.transformer_blocks)
        for i, block in enumerate(pipeline.transformer.transformer_blocks):
            processor = block.attn.processor
            if isinstance(processor, QwenDoubleStreamAttnProcessor2_0WithStyleControl):
                # 检查 processor 是否为 nn.Module
                if not isinstance(processor, nn.Module):
                    raise TypeError(f"Block {i} 的 Processor 不是 nn.Module！请修改 QwenDoubleStreamAttnProcessor2_0WithStyleControl 继承 nn.Module。")
                
                # 解冻并收集参数
                for param_name, param in processor.named_parameters():
                    if "style_k_proj" in param_name or "style_v_proj" in param_name:
                        param.requires_grad = True
                        trainable_params.append(param)
                        logger.info(f"✅ 解冻参数: block_{i}.{param_name}")
            else:
                logger.warning(f"Block {i} 的 processor 类型不匹配: {type(processor)}")
    
    if not trainable_params:
        raise ValueError("未找到任何可训练的 'style_k_proj' 或 'style_v_proj' 参数。请检查你的 Processor 实现。")
    
    logger.info(f"✨ 成功解冻 {len(trainable_params)} 个参数张量 (来自 {total_blocks} 个 blocks)。")
    logger.info(f"💰 可训练参数量: {sum(p.numel() for p in trainable_params) / 1_000_000:.2f} M")

    # 移动到设备 (在 prepare 之前)
    pipeline.to(accelerator.device)

    # --- 3. 准备 VAE 和 Scheduler (用于训练循环) ---
    # (vae 和 scheduler 已经包含在 pipeline 中, 并且被冻结)
    vae = pipeline.vae
    noise_scheduler = pipeline.scheduler
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    # VAE scale factor
    vae_scale_factor = 2 ** len(vae.temperal_downsample)
    
    # 从 VAE 配置中获取
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, 1, vae.config.z_dim, 1, 1)
        .to(accelerator.device, dtype=weight_dtype)
    )
    latents_std_inv = 1.0 / torch.tensor(vae.config.latents_std).view(1, 1, vae.config.z_dim, 1, 1).to(
        accelerator.device, dtype=weight_dtype
    )

    # (来自参考脚本)
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # --- 4. 优化器 ---
    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        trainable_params, # 只优化我们解冻的参数
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # --- 5. 数据集和 Dataloader ---
    train_dataset = StyleEditDataset(train_data_dir=args.data_config.train_data_dir)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # --- 6. 学习率调度器 ---
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # --- 7. Accelerator Prepare ---
    # 我们准备整个 pipeline，因为它包含了可训练的子模块
    pipeline, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        pipeline, optimizer, train_dataloader, lr_scheduler
    )

    # --- 8. 训练循环 ---
    global_step = 0
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = (steps: {args.max_train_steps})")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=0,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # 需要 pipeline 的 unwrap 版本来调用非 forward 方法
    unwrapped_pipeline = accelerator.unwrap_model(pipeline)
    
    # 固定 VAE 和 Text Encoder 为评估模式
    unwrapped_pipeline.vae.eval()
    unwrapped_pipeline.text_encoder.eval()
    # Transformer 主干被冻结，但包含可训练子模块，应处于 train 模式
    unwrapped_pipeline.transformer.train()

    # (来自你的 pipeline __call__)
    # 假设这些常量在导入时已定义
    # 如果没有，你需要在这里或 config 中定义它们
    _CONDITION_IMAGE_SIZE = CONDITION_IMAGE_SIZE if "CONDITION_IMAGE_SIZE" in globals() else 1024*1024
    _VAE_IMAGE_SIZE = VAE_IMAGE_SIZE if "VAE_IMAGE_SIZE" in globals() else 1024*1024

    for epoch in range(args.num_train_epochs): # 假设 config 中有 num_train_epochs
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            
            # --- 8.1 准备数据 (在 no_grad 上下文中) ---
            with torch.no_grad():
                bsz = len(batch["prompts"])
                device = accelerator.device
                
                # --- a) 目标 (Ground Truth) Latents ---
                gt_images_pil = batch["gt_images_pil"]
                gt_pixel_values_list = []
                for img in gt_images_pil:
                    w, h = calculate_dimensions(_VAE_IMAGE_SIZE, img.size[0] / img.size[1])
                    # preprocess 返回 (C, H, W)，需要 (C, 1, H, W)
                    gt_pixel_values_list.append(
                        unwrapped_pipeline.image_processor.preprocess(img, h, w).unsqueeze(2)
                    )
                gt_pixel_values = torch.cat(gt_pixel_values_list, dim=0).to(device, dtype=weight_dtype)
                
                # pixel_latents: (B, C, 1, H_lat, W_lat)
                pixel_latents = unwrapped_pipeline.vae.encode(gt_pixel_values).latent_dist.sample()
                pixel_latents = (pixel_latents - latents_mean) * latents_std_inv # 归一化
                
                target_height, target_width = pixel_latents.shape[3:]

                # --- b) 噪声和 Timesteps ---
                noise = torch.randn_like(pixel_latents, device=device, dtype=weight_dtype)
                u = compute_density_for_timestep_sampling(
                    weighting_scheme="none", # (来自参考脚本)
                    batch_size=bsz,
                    logit_mean=0.0,
                    logit_std=1.0,
                    mode_scale=1.29,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=device)
                
                sigmas = get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
                
                # (B, C, 1, H_lat, W_lat)
                noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise
                
                # Pack: (B, L_noise, C_packed)
                packed_noisy_model_input = unwrapped_pipeline._pack_latents(
                    noisy_model_input,
                    bsz, 
                    noisy_model_input.shape[2],
                    target_height,
                    target_width,
                )
                L_noise = packed_noisy_model_input.shape[1]

                # --- c) 文本条件 ---
                content_images_pil = batch["content_images_pil"]
                style_images_pil = batch["style_images_pil"]
                
                condition_images = [] # 用于 text encoder
                for img_list in zip(content_images_pil, style_images_pil):
                    img_pair = []
                    for img in img_list:
                        w, h = calculate_dimensions(_CONDITION_IMAGE_SIZE, img.size[0] / img.size[1])
                        img_pair.append(unwrapped_pipeline.image_processor.resize(img, h, w))
                    condition_images.append(img_pair[0]) # 假设 encode_prompt 只需 content
                    condition_images.append(img_pair[1]) # 假设 encode_prompt 需要 [content, style]
                
                # 你的 encode_prompt 接受 image 列表
                # (B, L_txt, D_txt)
                prompt_embeds, prompt_embeds_mask = unwrapped_pipeline.encode_prompt(
                    image=condition_images, # [content1, style1, content2, style2, ...] ?
                    prompt=batch["prompts"],
                    device=device,
                    num_images_per_prompt=1,
                    max_sequence_length=unwrapped_pipeline.tokenizer_max_length,
                )
                txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()

                # --- d) 图像/风格 条件 ---
                vae_images_list_content = []
                vae_images_list_style = []
                
                img_shapes_list = [] # 用于 transformer

                for content_img, style_img in zip(content_images_pil, style_images_pil):
                    # Content
                    w_c, h_c = calculate_dimensions(_VAE_IMAGE_SIZE, content_img.size[0] / content_img.size[1])
                    vae_images_list_content.append(
                        unwrapped_pipeline.image_processor.preprocess(content_img, h_c, w_c).unsqueeze(2)
                    )
                    # Style
                    w_s, h_s = calculate_dimensions(_VAE_IMAGE_SIZE, style_img.size[0] / style_img.size[1])
                    vae_images_list_style.append(
                        unwrapped_pipeline.image_processor.preprocess(style_img, h_s, w_s).unsqueeze(2)
                    )
                    
                    # (H_lat, W_lat)
                    noise_shape = (target_height // 2, target_width // 2)
                    content_shape = (h_c // (vae_scale_factor * 2) // 2, w_c // (vae_scale_factor * 2) // 2)
                    style_shape = (h_s // (vae_scale_factor * 2) // 2, w_s // (vae_scale_factor * 2) // 2)
                    
                    # (来自参考脚本，但为3部分调整)
                    img_shapes_list.append([(1, *noise_shape), (1, *content_shape), (1, *style_shape)])


                vae_images_content = torch.cat(vae_images_list_content, dim=0).to(device, dtype=weight_dtype)
                vae_images_style = torch.cat(vae_images_list_style, dim=0).to(device, dtype=weight_dtype)

                # 编码并 Pack
                content_latents = unwrapped_pipeline._encode_vae_image(vae_images_content, generator=None)
                style_latents_unpacked = unwrapped_pipeline._encode_vae_image(vae_images_style, generator=None)
                
                packed_content_latents = unwrapped_pipeline._pack_latents(
                    content_latents, bsz, content_latents.shape[1], content_latents.shape[3], content_latents.shape[4]
                )
                packed_style_latents = unwrapped_pipeline._pack_latents(
                    style_latents_unpacked, bsz, style_latents_unpacked.shape[1], style_latents_unpacked.shape[3], style_latents_unpacked.shape[4]
                )
                
                # (B, L_content + L_style, C_packed)
                image_latents = torch.cat([packed_content_latents, packed_style_latents], dim=1)
                
                L_content_patches = packed_content_latents.shape[1]
                L_style_patches = packed_style_latents.shape[1]
                
                # --- e) 准备 Kwargs ---
                attention_kwargs = {
                    "style_image_latents": packed_style_latents,
                    "style_start_idx": L_noise + L_content_patches,
                    "style_end_idx": L_noise + L_content_patches + L_style_patches,
                    "noise_patches_length": L_noise,
                    "content_patches_length": L_content_patches,
                    "style_scale": args.style_scale # 从 config 获取
                }
                
                # --- f) 最终输入 ---
                # (B, L_noise + L_content + L_style, C_packed)
                latent_model_input = torch.cat([packed_noisy_model_input, image_latents], dim=1)

            # --- 8.2 训练步骤 (开启梯度) ---
            with accelerator.accumulate(pipeline):
                # (B, L_total, C_packed)
                model_pred = unwrapped_pipeline.transformer(
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
                
                # --- 8.3 Loss 计算 ---
                # 只取噪声部分的预测
                model_pred = model_pred[:, :L_noise]
                
                # Unpack
                model_pred = unwrapped_pipeline._unpack_latents(
                    model_pred,
                    height=target_height * vae_scale_factor,
                    width=target_width * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )
                
                # (来自参考脚本)
                weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
                
                # Flow-matching loss: 目标是 (noise - pixel_latents)
                target = noise - pixel_latents
                
                # (B, C, 1, H, W) -> (B, 1, C, H, W)
                target = target.permute(0, 2, 1, 3, 4) 
                
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()
                
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # 裁剪可训练参数的梯度
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # --- 8.4 日志和检查点 ---
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss, "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        
                        # 只保存我们训练的参数
                        current_unwrapped_pipeline = accelerator.unwrap_model(pipeline)
                        style_control_state_dict = {}
                        
                        # 从 unwrapped模型 中提取权重
                        for name, param in current_unwrapped_pipeline.named_parameters():
                            if param.requires_grad:
                                # key 应该匹配加载时的 key
                                # e.g., 'transformer.transformer_blocks.0.attn.processor.style_k_proj.weight'
                                style_control_state_dict[name] = param.cpu().to(torch.float32).detach()
                        
                        if style_control_state_dict:
                            torch.save(style_control_state_dict, os.path.join(save_path, "style_control_weights.pth"))
                            logger.info(f"✅ Saved style control weights to {save_path}/style_control_weights.pth")
                        else:
                            logger.warning("未找到可保存的已训练权重。")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
        
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    
    # --- 9. 保存最终模型 ---
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "final_model")
        os.makedirs(save_path, exist_ok=True)
        
        current_unwrapped_pipeline = accelerator.unwrap_model(pipeline)
        style_control_state_dict = {}
        for name, param in current_unwrapped_pipeline.named_parameters():
            if param.requires_grad:
                style_control_state_dict[name] = param.cpu().to(torch.float32).detach()
        
        if style_control_state_dict:
            torch.save(style_control_state_dict, os.path.join(save_path, "style_control_weights.pth"))
            logger.info(f"✅ Saved final style control weights to {save_path}/style_control_weights.pth")

    accelerator.end_training()


if __name__ == "__main__":
    main()