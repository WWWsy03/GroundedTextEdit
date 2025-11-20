import argparse
import gc
import logging
import os
import math
import shutil
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.models import AutoencoderKLQwenImage, QwenImageTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor

# --- 导入你的自定义类 ---
# 假设你把修改后的 Pipeline 和 Processor 定义在 my_pipeline.py 中
# 如果没有文件，请将你的类定义直接粘贴到此处上方
from style_transfer_pipeline import QwenImageEditPlusPipelineWithStyleControl
from style_transfer_processor import QwenDoubleStreamAttnProcessor2_0WithStyleControl

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for Qwen Style Processor.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="/app/cold1/Qwen-Image-Edit-2509", help="Path to pretrained model")
    parser.add_argument("--data_dir", type=str, default="/app/code/texteditRoPE/train_data_dir", help="Root directory of training data")
    parser.add_argument("--output_dir", type=str, default="/app/code/texteditRoPE/qwenimage-style-control-output", help="Output directory")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--cache_dir", type=str, default="/app/code/texteditRoPE/ /cached_embeddings", help="Directory to save precomputed embeddings")
    parser.add_argument("--skip_cache", default=False,action="store_true", help="If true, skip cache generation and strictly load from existing cache")
    return parser.parse_args()

# --- 辅助函数 ---
def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return width, height

def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return latents

# --- 原始图片数据集 (用于生成缓存) ---
class RawImageDataset(Dataset):
    def __init__(self, root_dir, size=1024):
        self.root_dir = Path(root_dir)
        self.content_dir = self.root_dir / "content_images"
        self.style_dir = self.root_dir / "style_images"
        self.gt_dir = self.root_dir / "ground_truth_images"
        self.prompts_file = self.root_dir / "prompts.txt"
        self.size = size
        
        with open(self.prompts_file, "r", encoding="utf-8") as f:
            self.prompts = [line.strip() for line in f.readlines() if line.strip()]
            
        self.content_images = sorted([f for f in os.listdir(self.content_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        self.style_images = sorted([f for f in os.listdir(self.style_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        self.gt_images = sorted([f for f in os.listdir(self.gt_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

        # 确保数量一致
        assert len(self.content_images) == len(self.style_images) == len(self.gt_images) == len(self.prompts)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        # 返回原始 PIL 图片和 Prompt，具体的 Transform 交给 Pipeline 处理
        return {
            "content_img": Image.open(self.content_dir / self.content_images[idx]).convert("RGB"),
            "style_img": Image.open(self.style_dir / self.style_images[idx]).convert("RGB"),
            "gt_img": Image.open(self.gt_dir / self.gt_images[idx]).convert("RGB"),
            "prompt": self.prompts[idx],
            "id": idx # 用于命名缓存文件
        }

# --- 缓存数据集 (用于训练) ---
class CachedDataset(Dataset):
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.files = sorted(list(self.cache_dir.glob("*.pt")), key=lambda x: int(x.stem))
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location="cpu")
        return data

def main():
    args = parse_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=ProjectConfiguration(project_dir=args.output_dir)
    )
    
    logging.basicConfig(level=logging.INFO)
    logger.info(accelerator.state)
    
    # 设置缓存目录
    cache_dir = os.path.join(args.output_dir, args.cache_dir)
    
    # =====================================================
    # Phase 1: 数据预处理与缓存 (Pre-computation)
    # =====================================================
    if not args.skip_cache and accelerator.is_main_process:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            logger.info(f"Created cache directory: {cache_dir}")
            
            # 1. 加载完整 Pipeline
            logger.info("🚀 Loading Pipeline for pre-computation...")
            dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
            
            # 使用你的自定义 Pipeline 加载
            pipeline = QwenImageEditPlusPipelineWithStyleControl.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=dtype
            )
            pipeline.to(accelerator.device)
            pipeline.set_progress_bar_config(disable=True)

            # 2. 准备数据
            raw_dataset = RawImageDataset(args.data_dir, size=args.resolution)
            # 注意：预处理不使用 Dataloader 的 batch，为了简单起见逐个处理，防止 OOM
            # 且 Pipeline 的 encode_prompt 内部逻辑比较复杂，逐个处理更稳妥
            
            logger.info("💾 Starting pre-computation of embeddings and latents...")
            
            for idx in tqdm(range(len(raw_dataset)), desc="Caching Data"):
                item = raw_dataset[idx]
                
                # A. 计算尺寸
                # Pipeline 内部通常会 Resize，但我们需要确保输入 VAE 的尺寸是规整的
                # 这里我们手动 resize 到 target resolution
                # 注意：Qwen 需要 height/width 必须是 patch size 的倍数
                w, h = item["content_img"].size
                target_w, target_h = calculate_dimensions(args.resolution * args.resolution, w / h)
                
                # 使用 Pipeline 的 image_processor 可能会更方便，但手动控制更明确
                # Resize images for VAE (Latent Generation)
                resize_tf = transforms.Compose([
                    transforms.Resize((target_h, target_w), interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ])
                
                gt_pixel_values = resize_tf(item["gt_img"]).unsqueeze(0).to(pipeline.device, dtype=dtype)
                content_pixel_values = resize_tf(item["content_img"]).unsqueeze(0).to(pipeline.device, dtype=dtype)
                style_pixel_values = resize_tf(item["style_img"]).unsqueeze(0).to(pipeline.device, dtype=dtype)

                # Resize content image for Text Encoder (Qwen-VL Vision input)
                # Pipeline 内部的 _get_qwen_prompt_embeds 会调用 processor 处理，但我们需要传入合适的 PIL
                # 这里直接传 resize 后的 PIL 即可
                prompt_image_pil = item["content_img"].resize((target_w, target_h))

                with torch.no_grad():
                    # B. VAE Encoding (生成 Latents)
                    # 1. GT Latents
                    gt_latents = pipeline.vae.encode(gt_pixel_values).latent_dist.sample()
                    # 2. Content Latents
                    content_latents = pipeline.vae.encode(content_pixel_values).latent_dist.sample()
                    # 3. Style Latents
                    style_latents = pipeline.vae.encode(style_pixel_values).latent_dist.sample()
                    
                    # Normalize latents
                    latents_mean = torch.tensor(pipeline.vae.config.latents_mean).view(1, -1, 1, 1).to(pipeline.device, dtype=dtype)
                    latents_std = torch.tensor(pipeline.vae.config.latents_std).view(1, -1, 1, 1).to(pipeline.device, dtype=dtype)
                    
                    gt_latents = (gt_latents - latents_mean) / latents_std
                    content_latents = (content_latents - latents_mean) / latents_std
                    style_latents = (style_latents - latents_mean) / latents_std

                    # C. Text Encoding (生成 Prompt Embeddings)
                    # 调用 Pipeline 的方法。注意：Qwen-VL 需要 content image 作为 visual prompt
                    # condition_images 列表格式
                    prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
                        prompt=[item["prompt"]],
                        image=[prompt_image_pil], # 传入 PIL，pipeline 会处理
                        device=pipeline.device,
                        num_images_per_prompt=1,
                        max_sequence_length=1024 
                    )
                    
                    # D. 保存到磁盘 (CPU Tensor)
                    save_data = {
                        "gt_latents": gt_latents.cpu(),
                        "content_latents": content_latents.cpu(),
                        "style_latents": style_latents.cpu(),
                        "prompt_embeds": prompt_embeds.cpu(), # [1, seq_len, dim]
                        "prompt_embeds_mask": prompt_embeds_mask.cpu(),
                        "height": target_h,
                        "width": target_w
                    }
                    torch.save(save_data, os.path.join(cache_dir, f"{idx}.pt"))

            # 3. 清理显存
            logger.info("🗑️ Cleaning up Pipeline to free VRAM...")
            del pipeline
            # del vae # pipeline 里面有 vae，上面引用的是 pipeline.vae
            # del text_encoder
            gc.collect()
            torch.cuda.empty_cache()
        else:
            logger.info(f"Cache directory {cache_dir} already exists. Skipping generation.")

    # 等待主进程完成缓存
    accelerator.wait_for_everyone()

    # =====================================================
    # Phase 2: 训练设置 (Training Setup)
    # =====================================================
    logger.info("🚀 Starting Training Phase...")

    # 1. 只加载 Transformer (DiT) 和 Scheduler
    # VAE 和 Text Encoder 不需要了，因为数据已经是 Embeddings/Latents 了
    transformer = QwenImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    transformer.requires_grad_(False) # 冻结所有

    # 2. 替换 Attention Processor
    # 注意：我们需要参数来初始化 Processor，虽然 VAE 没加载，但维度是固定的
    # Qwen VAE z_dim = 16 (通常)
    latent_channels = 16 
    style_context_dim = latent_channels * 4 # 64 (packed)
    style_hidden_dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim

    logger.info("🔥 Swapping Attention Processors...")
    trainable_params = []
    for i, block in enumerate(transformer.transformer_blocks):
        new_processor = QwenDoubleStreamAttnProcessor2_0WithStyleControl(
            style_context_dim=style_context_dim,
            style_hidden_dim=style_hidden_dim
        )
        block.attn.processor = new_processor
        
        # 激活 style proj 参数
        for name, param in block.attn.processor.named_parameters():
            if "style_k_proj" in name or "style_v_proj" in name:
                param.requires_grad = True
                trainable_params.append(param)
    
    logger.info(f"✅ Processors swapped. Trainable params: {len(trainable_params)}")

    # 3. 优化器
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    # 4. 加载缓存数据集
    cached_dataset = CachedDataset(cache_dir)
    # 注意：dataset 返回的是 Tensor，DataLoader 会自动 collate
    dataloader = DataLoader(cached_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4)

    # 5. Prepare
    transformer, optimizer, dataloader = accelerator.prepare(transformer, optimizer, dataloader)
    
    # 训练循环
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.max_train_steps
    num_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    global_step = 0

    # 辅助函数获取 sigmas (因为 helper pipeline 没了，我们需要手动调用 scheduler)
    def get_sigmas(timesteps, n_dim, dtype, device):
        sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    for epoch in range(num_epochs):
        transformer.train()
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(transformer):
                # Batch 中的数据已经是 Latents/Embeddings
                # batch["gt_latents"]: [B, 16, H, W]
                print(f"batch keys: {batch.keys()}")
                gt_latents = batch["gt_latents"].to(dtype=transformer.dtype)
                content_latents = batch["content_latents"].to(dtype=transformer.dtype)
                style_latents = batch["style_latents"].to(dtype=transformer.dtype)
                prompt_embeds = batch["prompt_embeds"].to(dtype=transformer.dtype) # [B, 1, Seq, Dim] -> [B, Seq, Dim]
                prompt_embeds_mask = batch["prompt_embeds_mask"] # [B, 1, Seq] -> [B, Seq]

                # 处理维度 squeeze (因为 precompute 保存时可能是 [1, ...])
                if prompt_embeds.ndim == 4:
                    prompt_embeds = prompt_embeds.squeeze(1)
                if prompt_embeds_mask.ndim == 3:
                    prompt_embeds_mask = prompt_embeds_mask.squeeze(1)

                bsz = gt_latents.shape[0]
                # 从 batch 中获取高度宽度 (假设 batch 内尺寸一致，通常是)
                height = batch["height"][0].item()
                width = batch["width"][0].item()

                # --- Pack Latents ---
                # 此时 latents 还是 [B, 16, H/8, W/8] (unpacked)
                # 需要 pack 成 Qwen 输入格式
                packed_gt = _pack_latents(gt_latents, bsz, latent_channels, height, width)
                packed_content = _pack_latents(content_latents, bsz, latent_channels, height, width)
                packed_style = _pack_latents(style_latents, bsz, latent_channels, height, width)

                L_noise = packed_gt.shape[1]
                L_content = packed_content.shape[1]
                L_style = packed_style.shape[1]

                # --- Add Noise ---
                noise = torch.randn_like(packed_gt)
                u = compute_density_for_timestep_sampling(
                    weighting_scheme="none", batch_size=bsz, logit_mean=0.0, logit_std=1.0, mode_scale=1.29
                )
                indices = (u * scheduler.config.num_train_timesteps).long()
                timesteps = scheduler.timesteps[indices].to(device=accelerator.device)
                
                sigmas = get_sigmas(timesteps, n_dim=packed_gt.ndim, dtype=packed_gt.dtype, device=accelerator.device)
                noisy_model_input = (1.0 - sigmas) * packed_gt + sigmas * noise

                # --- Concat ---
                latent_model_input = torch.cat([noisy_model_input, packed_content, packed_style], dim=1)

                # --- RoPE Shapes ---
                # 构造形状 [ (1, h/16, w/16), ... ]
                patch_h = height // 2 // 8  # VAE factor 8, patch factor 2
                patch_w = width // 2 // 8   # 其实可以直接用 latent_h // 2
                # 更准确的计算方式，直接用 packed_gt 的形状推算
                # packed 形状: [B, (H/16)*(W/16), C*4]
                # Qwen RoPE 需要 (1, H_grid, W_grid)
                # VAE Latent H = H_orig / 8.
                # Grid H = Latent H / 2.
                grid_h = height // 16
                grid_w = width // 16
                rope_shape = (1, grid_h, grid_w)
                img_shapes = [[rope_shape, rope_shape, rope_shape]] * bsz

                # --- Attention Kwargs ---
                style_start_idx = L_noise + L_content
                style_end_idx = style_start_idx + L_style
                txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()

                attention_kwargs = {
                    "style_image_latents": packed_style,
                    "style_start_idx": style_start_idx,
                    "style_end_idx": style_end_idx,
                    "noise_patches_length": L_noise,
                    "content_patches_length": L_content,
                    "style_scale": 1.0
                }

                # --- Forward ---
                model_pred = transformer(
                    hidden_states=latent_model_input,
                    timestep=timesteps / 1000,
                    guidance=None,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                # --- Loss ---
                model_pred = model_pred[:, :L_noise]
                target = noise - packed_gt
                
                weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % 10 == 0:
                    logger.info(f"Step {global_step}, Loss: {loss.item()}")

                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    
                    unwrapped_model = accelerator.unwrap_model(transformer)
                    style_weights = {}
                    for name, param in unwrapped_model.named_parameters():
                        if "style_k_proj" in name or "style_v_proj" in name:
                            style_weights[name] = param.detach().cpu()
                    
                    torch.save(style_weights, os.path.join(save_path, "style_adapter.pt"))
                    logger.info(f"Saved style adapter to {save_path}")

            if global_step >= max_train_steps:
                break

    accelerator.end_training()

if __name__ == "__main__":
    main()