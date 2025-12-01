from diffusers import DiffusionPipeline
import torch
import json
import os
import random
import argparse # 新增：用于解析命令行参数

# === 配置 ===
STYLE_FILE = "/app/cold1/code/texteditRoPE/data_construction_style/styles_corpus.json"
OUTPUT_IMG_DIR = "dataset_images_3"
OUTPUT_STYLE_IMG_DIR = "dataset_images_3/style_images"
OUTPUT_GT_IMG_DIR = "dataset_images_3/images"
MODEL_NAME = "/app/cold1/Qwen-Image" 

# 扩展了中英文词表，包含不同长度和类型的词
from assets.word_list import WORD_LIST

# === 位置和倾斜度的描述列表 ===
POSITIONS = [
    "centered in the frame", 
    "positioned at the top center",
    "positioned at the bottom center",
    "positioned on the left side",
    "positioned on the right side",
    "located in the top-left corner",
    "located in the top-right corner",
    "located in the bottom-left corner",
    "located in the bottom-right corner",
    "slightly off-center to the left",
    "slightly off-center to the top"
]

TILTS = [
    "straight orientation, no tilt", 
    "straight orientation, no tilt",
    "tilted slightly clockwise",
    "tilted slightly counter-clockwise",
    "rotated about 15 degrees clockwise",
    "rotated about 15 degrees counter-clockwise",
    "rotated about 45 degrees clockwise",
    "rotated about 45 degrees counter-clockwise",
]

# 确保所有进程都创建目录
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_STYLE_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_GT_IMG_DIR, exist_ok=True)

def generate_image_pairs(rank, world_size):
    """
    rank: 当前进程的编号 (例如 0, 1, 2, 3)
    world_size: 总进程数 (例如 4)
    """
    print(f"--- Step 2: Generating Image Pairs [Worker {rank}/{world_size}] ---")
    
    with open(STYLE_FILE, 'r', encoding='utf-8') as f:
        styles = json.load(f)

    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device = "cuda"
    else:
        torch_dtype = torch.float32
        device = "cpu"

    # 加载文生图模型
    # 注意：如果是多卡单机运行，建议在外部通过 CUDA_VISIBLE_DEVICES 控制，
    # 或者在这里根据 rank 指定 device，例如 device_map={"": rank} (但这需要 rank 对应 gpu id)
    # 这里保持 balanced，依赖外部环境变量控制可见显卡
    pipe = DiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch_dtype, device_map="balanced")
    
    dataset_metadata = []

    # === 修改核心逻辑：只处理属于当前 rank 的数据 ===
    total_styles = len(styles)
    
    for idx, style_desc in enumerate(styles):
        # 模运算筛选：只处理 idx % world_size == rank 的数据
        #if idx <720: continue  # 测试时跳过前720个
        if idx % world_size != rank:
            continue

        # 1. 准备文本内容
        word_ref = random.choice(WORD_LIST)
        available_targets = [w for w in WORD_LIST if w != word_ref and len(w) > 1]
        if not available_targets: available_targets = ["Target"]
        word_target = random.choice(available_targets)
        
        # 2. 准备空间参数
        target_pos = random.choice(POSITIONS)
        target_tilt = random.choice(TILTS)

        # 3. 构造 Prompts
        base_template = 'The word "{}" rendered in {} style on a pure white background, high quality, detailed 3d render, cinematic lighting, {}.'
        prompt_ref = base_template.format(word_ref, style_desc, "centered, straight orientation")
        spatial_desc = f"{target_pos}, {target_tilt}"
        prompt_target = base_template.format(word_target, style_desc, spatial_desc)
        
        # 4. 设置 Seed
        seed = random.randint(0, 2**32 - 1)
        
        print(f"[Worker {rank}] Processing {idx}/{total_styles}: {style_desc[:20]}...")

        # 生成参考图
        generator_ref = torch.Generator(device=device).manual_seed(seed)
        image_ref = pipe(
            prompt=prompt_ref,
            height=1024, width=1024,
            num_inference_steps=30, 
            generator=generator_ref
        ).images[0]
        
        # 生成目标图
        generator_target = torch.Generator(device=device).manual_seed(seed)
        image_target = pipe(
            prompt=prompt_target,
            height=1024, width=1024,
            num_inference_steps=30,
            generator=generator_target
        ).images[0]

        # 5. 保存文件
        ref_filename = f"pair_{idx:05d}_ref.jpg"
        target_filename = f"pair_{idx:05d}_target.jpg"
        
        image_ref.save(os.path.join(OUTPUT_STYLE_IMG_DIR, ref_filename), quality=95)
        image_target.save(os.path.join(OUTPUT_GT_IMG_DIR, target_filename), quality=95)

        dataset_metadata.append({
            "pair_id": idx,
            "style_description": style_desc,
            "reference": {
                "word": word_ref,
                "image_path": os.path.join(OUTPUT_STYLE_IMG_DIR, ref_filename),
                "position": "centered",
                "tilt": "straight"
            },
            "target": {
                "word": word_target,
                "image_path": os.path.join(OUTPUT_GT_IMG_DIR, target_filename),
                "position_prompt": target_pos,
                "tilt_prompt": target_tilt
            },
            "generation_seed": seed
        })

        # === 修改：WIP 文件名带上 rank，避免多进程写冲突 ===
        if len(dataset_metadata) % 10 == 0:
             wip_filename = f"dataset_metadata_step2_wip_part{rank}.json"
             with open(wip_filename, "w", encoding='utf-8') as f:
                json.dump(dataset_metadata, f, indent=4, ensure_ascii=False)
             print(f"  -> [Worker {rank}] Saved WIP metadata")

    # === 修改：最终文件名带上 rank ===
    final_filename = f"dataset_metadata_step2_final_part{rank}.json"
    with open(final_filename, "w", encoding='utf-8') as f:
        json.dump(dataset_metadata, f, indent=4, ensure_ascii=False)
        
    print(f"--- Worker {rank} Done. Generated {len(dataset_metadata)} pairs. ---")
    return dataset_metadata

if __name__ == "__main__":
    # 确保有风格文件
    if not os.path.exists(STYLE_FILE):
        print(f"Error: Style file '{STYLE_FILE}' not found. Please run Step 1 first.")
    else:
        # 解析命令行参数
        parser = argparse.ArgumentParser()
        # 默认为 1，总数为 4 (对应你要求的 模4余1)
        parser.add_argument("--rank", type=int, default=0, help="Current worker ID (remainder)")
        parser.add_argument("--world_size", type=int, default=8, help="Total number of workers (divisor)")
        
        args = parser.parse_args()
        
        generate_image_pairs(args.rank, args.world_size)