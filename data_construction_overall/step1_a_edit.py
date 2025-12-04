import os
import json
import random
import importlib.util
import argparse
import torch
from PIL import Image
from diffusers import DiffusionPipeline

import config

# === 配置参数 ===
PROB_NO_TEXT = 0.1  # 15% 的概率不添加任何文字（原图直接作为结果）

# === 工具函数：加载词表 ===
def load_word_list_from_module(path):
    spec = importlib.util.spec_from_file_location("word_list_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WORD_LIST

def generate_edited_images():
    # 0. 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=3, help="当前进程编号")
    parser.add_argument("--world_size", type=int, default=4, help="总进程数")
    args = parser.parse_args()

    print(f"--- Step 1-A: Generating Edited Images [Rank {args.rank}/{args.world_size}] ---")
    
    # 1. 准备临时输出目录
    TEMP_EDIT_DIR = os.path.join(config.OUTPUT_DIR, "temp_edited_intermediate")
    os.makedirs(TEMP_EDIT_DIR, exist_ok=True)
    
    # 2. 加载资源
    word_list = load_word_list_from_module(config.WORD_LIST)
    bg_files = [f for f in os.listdir(config.RAW_BG_DIR) if f.endswith(('.jpg', '.png'))]
    
    if not bg_files:
        print(f"Error: No images found in {config.RAW_BG_DIR}")
        return

    # 3. 加载编辑模型
    # 只有当确实需要编辑时才加载模型，不过考虑到批处理中大概率会用到，这里直接加载
    print(f"Loading Edit Model: {config.MODEL_QWEN_PATH}")
    try:
        pipe = DiffusionPipeline.from_pretrained(
            config.MODEL_QWEN_PATH, 
            torch_dtype=torch.bfloat16, 
            device_map="balanced"
        )
    except Exception as e:
        print(f"Warning: Model loading failed (check path or GPU). Error: {e}")
        return

    intermediate_metadata = []
    
    # 4. 循环生成
    for i in range(config.NUM_SAMPLES):
        # === 分布式过滤：只处理属于当前 rank 的数据 ===
        if i % args.world_size != args.rank:
            continue

        bg_name = random.choice(bg_files)
        bg_path = os.path.join(config.RAW_BG_DIR, bg_name)
        
        # 决定是否添加文字
        # True: 不加字 (No Text); False: 加字 (Edit)
        is_empty_sample = random.random() < PROB_NO_TEXT
        
        if is_empty_sample:
            temp_word = ""  # 设为空
            print(f"[{i}/{config.NUM_SAMPLES}] Copying BG (No Text)...")
        else:
            temp_word = random.choice(word_list)
            print(f"[{i}/{config.NUM_SAMPLES}] Editing {bg_name} with '{temp_word}'...")

        try:
            # 准备原图
            original_image = Image.open(bg_path).convert("RGB").resize(config.IMAGE_SIZE)
            
            if is_empty_sample:
                # === 分支 A: 不加文字 ===
                # 直接使用原图作为 edited_img
                edited_img = original_image
            else:
                # === 分支 B: 调用模型加文字 ===
                instruction = f"在图中合适位置绘制艺术字“{temp_word}”，不改变图片原本样子"
                
                generator = torch.Generator(device="cuda").manual_seed(random.randint(0, 100000))
                edited_img = pipe(
                    prompt=instruction, 
                    image=original_image, 
                    num_inference_steps=30,
                    generator=generator
                ).images[0]
            
            # 保存编辑后的中间图 (如果是空样本，这里存的就是resize后的原图)
            temp_filename = f"{i:06d}_edited_temp.jpg"
            temp_path = os.path.join(TEMP_EDIT_DIR, temp_filename)
            edited_img.save(temp_path)
            
            # 记录中间元数据
            intermediate_metadata.append({
                "id": i,
                "original_bg_path": bg_path,   
                "edited_temp_path": temp_path, 
                "temp_word_used": temp_word,   # 可能是空字符串
                "bg_name": bg_name
            })
            
        except Exception as e:
            print(f" -> Error processing sample {i}: {e}")
            continue

    # 保存中间 Json，文件名带上 rank 防止冲突
    json_filename = f"step1_intermediate_rank{args.rank}.json"
    intermediate_json_path = os.path.join(config.SUB_DIRS["meta"], json_filename)
    
    with open(intermediate_json_path, "w", encoding='utf-8') as f:
        json.dump(intermediate_metadata, f, indent=4, ensure_ascii=False)
        
    print(f"\n--- Step 1-A (Rank {args.rank}) Complete ---")
    print(f"Metadata saved to: {intermediate_json_path}")

if __name__ == "__main__":
    generate_edited_images()