import argparse
from accelerate.logging import get_logger
import os
import torch
from pathlib import Path
from tqdm.auto import tqdm
from PIL import Image
from accelerate import Accelerator
from omegaconf import OmegaConf

# 导入你的 Pipeline 和辅助函数
from style_transfer_pipeline import QwenImageEditPlusPipelineWithStyleControl, calculate_dimensions, CONDITION_IMAGE_SIZE
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models import AutoencoderKLQwenImage, QwenImageTransformer2DModel
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor

def parse_args():
    parser = argparse.ArgumentParser(description="Pre-compute embeddings for Qwen-Image-Edit.")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    return parser.parse_args()

def main():
    args = argparse.Namespace(**OmegaConf.load(parse_args().config))
    
    # 定义缓存目录
    cache_dir = Path(args.data_config.train_data_dir) / "cached_embeddings"
    cache_dir.mkdir(exist_ok=True)
    
    accelerator = Accelerator()
    
    # 1. 加载完整的模型用于计算 (只需要 Text Encoder 相关，但为了 pipeline 逻辑兼容，我们加载需要的)
    # VAE 和 Transformer 其实不需要加载到 GPU，但为了 pipeline 初始化不报错，传个空的或 cpu 的也行
    # 这里为了简单，我们正常加载 Text Encoder 到 GPU
    
    logger = get_logger(__name__, log_level="INFO")
    logger.info("Loading Text Encoder and Pipeline for pre-computation...")

    weight_dtype = torch.bfloat16 # 推荐用 bf16 计算
    
    # 我们只需要 Text Encoder 相关的
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=weight_dtype
    )
    tokenizer = Qwen2Tokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    processor = Qwen2VLProcessor.from_pretrained(args.pretrained_model_name_or_path, subfolder="processor")
    
    # 为了让 Pipeline 跑起来，我们需要 mock 其他组件或者加载它们
    # 这里我们只把 TextEncoder 放到 GPU，其他留空或者 CPU
    pipeline = QwenImageEditPlusPipelineWithStyleControl(
        scheduler=FlowMatchEulerDiscreteScheduler(), # Dummy
        vae=None, # 不需要
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        processor=processor,
        transformer=None, # 不需要
    )
    pipeline.to(accelerator.device)

    # 2. 准备数据列表
    data_dir = Path(args.data_config.train_data_dir)
    content_dir = data_dir / "content_images" # 假设你的 config 没变
    style_dir = data_dir / "style_images"
    prompt_file = data_dir / "prompts.txt"
    
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    
    # 简单的文件匹配逻辑 (假设文件名排序对应)
    content_files = sorted([f for f in content_dir.glob('*') if f.suffix.lower() in ['.jpg', '.png']])
    style_files = sorted([f for f in style_dir.glob('*') if f.suffix.lower() in ['.jpg', '.png']])

    logger.info(f"Found {len(prompts)} items. Starting pre-computation...")

    # 3. 循环计算并保存
    for i, prompt in enumerate(tqdm(prompts)):
        # 文件名用于保存 ID
        file_id = content_files[i].stem 
        save_path = cache_dir / f"{file_id}.pt"
        
        if save_path.exists():
            continue # 支持断点续传

        # 加载图片用于 Text Encoder 的输入
        c_img = Image.open(content_files[i]).convert("RGB")
        s_img = Image.open(style_files[i]).convert("RGB")
        
        # 预处理图片 (Resize)
        condition_images = []
        for img in [c_img, s_img]:
            w, h = calculate_dimensions(CONDITION_IMAGE_SIZE, img.size[0] / img.size[1])
            condition_images.append(pipeline.image_processor.resize(img, h, w))
        
        with torch.no_grad():
            # 调用 Pipeline 的 encode_prompt
            # 注意：这会返回 GPU 上的 tensor
            prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
                image=condition_images,
                prompt=[prompt],
                device=accelerator.device,
                num_images_per_prompt=1,
                max_sequence_length=1024 # 或 pipeline.tokenizer_max_length
            )
            
            # 保存到 CPU 硬盘
            torch.save({
                "prompt_embeds": prompt_embeds.cpu(), # [1, seq_len, dim]
                "prompt_embeds_mask": prompt_embeds_mask.cpu() # [1, seq_len]
            }, save_path)

    logger.info("✅ All embeddings cached successfully!")

if __name__ == "__main__":
    main()