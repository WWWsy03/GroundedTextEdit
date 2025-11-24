from diffusers import DiffusionPipeline
import torch
import json
import os
import random

# === 配置 ===
STYLE_FILE = "/app/code/texteditRoPE/data_construction_style/styles_corpus.json"
OUTPUT_IMG_DIR = "dataset_images"
OUTPUT_STYLE_IMG_DIR="dataset_images/style_images"
OUTPUT_GT_IMG_DIR="dataset_images/images"
MODEL_NAME = "/app/cold1/Qwen-Image" 

# 扩展了中英文词表，包含不同长度和类型的词
WORD_LIST = [
    "Knight", "吃烧烤", "Coffee", "Magic", "早上好啊！", "Neon", "Stone", "Water",
    "Dragon", "打麻将", "Tea", "Dream", "晚安zzz~", "Pixel", "Fire", "Wind",
    "Wizard", "撸猫", "Latte", "Chaos", "加油！", "Glitch", "Ice", "Cloud",
    "Samurai", "嗦粉", "Mocha", "Time", "哈哈哈", "Cyber", "Lava", "Rain",
    "Phoenix", "追剧", "Matcha", "Void", "冲鸭～", "Hologram", "Sand", "Fog",
    "Ninja", "逛夜市", "Espresso", "Echo", "稳了！", "Quantum", "Ash", "Dew",
    "Sorcerer", "贴秋膘", "Cappuccino", "Myth", "绝了！", "Synth", "Mist", "Wave",
    "Alchemist", "嗑瓜子", "Americano", "BBQ", "Future"
]

# === 新增：位置和倾斜度的描述列表 ===
# 位置描述
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

# 倾斜角度描述
TILTS = [
    "straight orientation, no tilt", # 大概率保持正向
    "straight orientation, no tilt",
    "straight orientation, no tilt",
    "tilted slightly clockwise",
    "tilted slightly counter-clockwise",
    "rotated about 15 degrees clockwise",
    "rotated about 15 degrees counter-clockwise",
]

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_STYLE_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_GT_IMG_DIR, exist_ok=True)

def generate_image_pairs():
    print("--- Step 2: Generating Image Pairs with Spatial Variation ---")
    
    with open(STYLE_FILE, 'r', encoding='utf-8') as f:
        styles = json.load(f)

    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device = "cuda"
    else:
        torch_dtype = torch.float32
        device = "cpu"

    # 加载文生图模型
    pipe = DiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch_dtype,device_map="balanced")
    #pipe = pipe.to(device)
    
    dataset_metadata = []

    for idx, style_desc in enumerate(styles):
        # 1. 准备文本内容
        word_ref = random.choice(WORD_LIST)
        # 确保目标词和参考词不一样，避免模型偷懒
        available_targets = [w for w in WORD_LIST if w != word_ref and len(w) > 1]
        if not available_targets: available_targets = ["Target"] # fallback
        word_target = random.choice(available_targets)
        
        # 2. 准备空间参数 (仅用于目标图)
        target_pos = random.choice(POSITIONS)
        target_tilt = random.choice(TILTS)

        # 3. 构造 Prompts
        # 基础模板，强调3D渲染和高质量
        base_template = 'The word "{}" rendered in {} style on a pure white background, high quality, detailed 3d render, cinematic lighting,  {}.'

        # --- 参考图 Prompt (保持稳定：居中，无倾斜) ---
        # 我们显式地加上 "centered, straight orientation"
        prompt_ref = base_template.format(word_ref, style_desc, "centered, straight orientation")
        
        # --- 目标图 Prompt (加入随机空间描述) ---
        spatial_desc = f"{target_pos}, {target_tilt}"
        prompt_target = base_template.format(word_target, style_desc, spatial_desc)
        
        # 4. 设置 Seed
        # 关键技巧：使用相同的 Seed 生成一对图。
        # 虽然 Prompt 变了（文字内容和位置），但相同的 Seed 有助于保持材质纹理、光照氛围和背景的一致性。
        seed = random.randint(0, 2**32 - 1)
        
        print(f"[{idx}/{len(styles)}] Style: {style_desc[:30]}...")
        print(f"  Ref: '{word_ref}' (Centered)")
        print(f"  Target: '{word_target}' ({target_pos}, {target_tilt})")

        # 生成参考图
        generator_ref = torch.Generator(device=device).manual_seed(seed)
        image_ref = pipe(
            prompt=prompt_ref,
            height=1024, width=1024,
            num_inference_steps=30, 
            generator=generator_ref
        ).images[0]
        
        # 生成目标图 (重新初始化 Generator 以确保使用相同的 Seed 开始)
        generator_target = torch.Generator(device=device).manual_seed(seed)
        image_target = pipe(
            prompt=prompt_target,
            height=1024, width=1024,
            num_inference_steps=30,
            generator=generator_target
        ).images[0]

        # 5. 保存文件和元数据
        # 使用更清晰的文件命名规范
        ref_filename = f"pair_{idx:04d}_ref.jpg"
        target_filename = f"pair_{idx:04d}_target.jpg"
        
        # 使用 JPEG 保存以节省空间，质量设为 95
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
                # 记录下具体的空间参数，这在后续分析或 Debug 时很有用
                "position_prompt": target_pos,
                "tilt_prompt": target_tilt
            },
            "generation_seed": seed
        })

        # 每生成 10 对就保存一次元数据，防止意外中断
        if (idx + 1) % 10 == 0:
             with open("dataset_metadata_step2_wip.json", "w", encoding='utf-8') as f:
                json.dump(dataset_metadata, f, indent=4, ensure_ascii=False)
             print(f"  -> Saved WIP metadata at index {idx}")

    # 保存最终完整元数据
    with open("dataset_metadata_step2_final.json", "w", encoding='utf-8') as f:
        json.dump(dataset_metadata, f, indent=4, ensure_ascii=False)
        
    print(f"--- Done. Generated {len(dataset_metadata)} pairs. ---")
    return dataset_metadata

if __name__ == "__main__":
    # 确保有风格文件
    if not os.path.exists(STYLE_FILE):
        print(f"Error: Style file '{STYLE_FILE}' not found. Please run Step 1 first.")
    else:
        generate_image_pairs()