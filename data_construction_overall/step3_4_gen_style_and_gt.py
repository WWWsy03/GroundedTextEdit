import os
import json
import random
import sys
import importlib.util
import argparse
import glob
import numpy as np
import torch
from PIL import Image
from diffusers import DiffusionPipeline, OvisImagePipeline

import config

# ==========================================
# 1. 核心组件：多样化指令生成器 (用于生成训练标签)
# ==========================================
def format_target_words(words_list):
    """
    将词列表格式化为自然语言字符串
    """
    if not words_list:
        return ""
    quoted_words = [f"“{w}”" for w in words_list]
    if len(quoted_words) == 1:
        return quoted_words[0]
    else:
        return "和".join(quoted_words)

def generate_diverse_instruction(remove_word, target_words_list):
    """
    根据是否有擦除词，以及目标词列表，随机生成多样化的指令
    (这是写入 groundtruth.txt 的训练用指令，不是给 Qwen 生成图片的 prompt)
    """
    draw_phrase = format_target_words(target_words_list)
    
    if not remove_word:
        templates = [
            f"参考最后一张图的风格，在图中绘制{draw_phrase}",
            f"使用参考图中的字体风格，在画面中添加文字{draw_phrase}",
            f"保持当前背景不变，用给定的风格写上{draw_phrase}",
            f"请在图中生成{draw_phrase}，字体风格请参照最后一张图",
            f"依据参考图的艺术风格，在图片合适位置植入{draw_phrase}",
            f"在图片空白处添加{draw_phrase}，风格参考最后一张图片",
            f"Draw {draw_phrase} using the style from the reference image"
        ]
    else:
        templates = [
            f"去除图中“{remove_word}”，并参考最后一张图中的文字风格在图中绘制{draw_phrase}",
            f"把“{remove_word}”删掉，然后用参考图风格重绘{draw_phrase}",
            f"将图片中的“{remove_word}”替换为{draw_phrase}，风格参考最后一张图",
            f"擦除文字“{remove_word}”，并在原布局位置用新风格写入{draw_phrase}",
            f"Remove '{remove_word}' and add {draw_phrase} in the reference style",
            f"请把“{remove_word}”改成{draw_phrase}，字体样式需与参考图一致",
            f"消除原有的“{remove_word}”，并在图中重新设计{draw_phrase}"
        ]
    
    return random.choice(templates)

def load_word_list_from_module(path):
    spec = importlib.util.spec_from_file_location("word_list_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WORD_LIST

# ==========================================
# 2. 主逻辑
# ==========================================
def process_step3_and_4():
    # === 0. 参数解析 ===
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0, help="Current process rank")
    parser.add_argument("--world_size", type=int, default=1, help="Total number of processes")
    args = parser.parse_args()

    print(f"=== Starting Combined Step 3 (Style) & Step 4 (GT) Generation [Rank {args.rank}/{args.world_size}] ===")
    
    # --- 准备数据 ---
    if not os.path.exists(config.STYLE_FILE):
        print(f"Error: Style file not found at {config.STYLE_FILE}")
        return
    with open(config.STYLE_FILE, 'r', encoding='utf-8') as f:
        styles_corpus = json.load(f)

    word_list = load_word_list_from_module(config.WORD_LIST)

    # 读取 Step 1 和 Step 2 元数据
    meta_dir = config.SUB_DIRS["meta"]
    try:
        step1_files = glob.glob(os.path.join(meta_dir, "step1_rank*.json"))
        if not step1_files: step1_files = glob.glob(os.path.join(meta_dir, "step1.json"))
        data1_dict = {}
        for fpath in step1_files:
            with open(fpath, "r", encoding='utf-8') as f:
                data = json.load(f)
                for item in data: data1_dict[item['id']] = item
            
        step2_files = glob.glob(os.path.join(meta_dir, "step2_rank*.json"))
        if not step2_files: step2_files = glob.glob(os.path.join(meta_dir, "step2.json"))
        step2_data = []
        for fpath in step2_files:
            with open(fpath, "r", encoding='utf-8') as f:
                step2_data.extend(json.load(f))
                
    except FileNotFoundError:
        print("Error: metadata files not found.")
        return

    print(f"Loaded {len(data1_dict)} Step 1 items and {len(step2_data)} Step 2 items.")

    processed_records = []
    
    # ------------------------------------------------------
    # PHASE 1: Generate Styles (Step 3) - Ovis
    # (保留此步骤以生成风格参考图，用于数据集构建)
    # ------------------------------------------------------
    print(f"\n--- Phase 1: Generating Style Reference Images (Rank {args.rank}) ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        pipe_style = OvisImagePipeline.from_pretrained(
            "/app/cold1/Ovis-Image", 
            torch_dtype=torch.bfloat16
        )
        pipe_style.to(device)
    except Exception as e:
        print(f"Error loading Ovis Pipeline: {e}")
        return
    
    for item in step2_data:
        idx = item['id']
        if idx % args.world_size != args.rank: continue
            
        assigned_style = random.choice(styles_corpus)
        style_ref_word = random.choice(word_list) 
        
        if idx not in data1_dict: continue
            
        print(f"[{idx}] Generating Style Ref: '{style_ref_word}' ({assigned_style})...")
        prompt_style = f'The word "{style_ref_word}" rendered in {assigned_style} style pure white background, high quality, detailed render, cinematic lighting.'
        
        try:
            gen = torch.Generator(device=device).manual_seed(random.randint(0, 100000))
            # style_img = pipe_style(
            #     prompt=prompt_style,
            #     height=1024, width=1024,
            #     num_inference_steps=30,
            #     generator=gen
            # ).images[0]
            
            save_name = f"{idx:06d}_style.jpg"
            save_path = os.path.join(config.SUB_DIRS["style"], save_name)
            #style_img.save(save_path)
            
            processed_records.append({
                "id": idx,
                "step2_item": item,
                "step1_item": data1_dict[idx],
                "assigned_style": assigned_style,
                "style_path": save_path,
                "style_ref_word": style_ref_word
            })
        except Exception as e:
            print(f" -> Style Gen Error: {e}")
            continue

    del pipe_style
    torch.cuda.empty_cache()

    # ------------------------------------------------------
    # PHASE 2: Generate GT (Step 4) & Instructions
    # ------------------------------------------------------
    print(f"\n--- Phase 2: Generating GT (Direct Generation) (Rank {args.rank}) ---")

    print(f"Loading Edit Model: {config.MODEL_QWEN_PATH}")
    # 假设该 Pipeline 支持 image列表 输入 [img1, img2]
    pipe_edit = DiffusionPipeline.from_pretrained(
        config.MODEL_QWEN_PATH, 
        torch_dtype=torch.bfloat16, 
        device_map="balanced" 
    )

    final_metadata = []
    gt_dir = os.path.join(config.OUTPUT_DIR, "groundtruth")
    os.makedirs(gt_dir, exist_ok=True)

    for record in processed_records:
        idx = record['id']
        temp_word = record['step1_item'].get('temp_word_used', "")
        target_word_str = record['step2_item']['target_word'] 
        target_words_list = record['step2_item'].get('target_words_list', [target_word_str])
        
        content_path = record['step2_item']['content_path'] # Step 2 的文字排版图
        style_desc = record['assigned_style']
        bg_path = record['step1_item']['input_path']       # Step 1 的背景图
        
        print(f"[{idx}] GT: '{bg_path}' + '{content_path}' -> {style_desc}")

        try:
            # 1. 准备输入图像
            bg_img = Image.open(bg_path).convert("RGB").resize(config.IMAGE_SIZE)
            content_img = Image.open(content_path).convert("RGB").resize(config.IMAGE_SIZE)

            # 2. 构造生成 Prompt
            # "在第一张图上把第二张图的文字用{风格描述字符串}风格绘制"
            prompt_gen = f"在第一张图上把第二张图的文字用{style_desc}风格绘制,保持文字位置不变"
            
            # 3. 生成 GT
            # 注意：这里传入一个图像列表 [bg, content]
            gen = torch.Generator(device="cuda").manual_seed(1234)
            final_gt = pipe_edit(
                prompt=prompt_gen,
                image=[bg_img, content_img], 
                num_inference_steps=30,
                generator=gen
            ).images[0]

            # 4. 保存 GT
            gt_filename = f"{idx:06d}_groundtruth.jpg"
            gt_save_path = os.path.join(gt_dir, gt_filename)
            final_gt.save(gt_save_path, quality=95)
            
            # 5. 生成多样化指令 (训练用的 text label)
            instruction_text = generate_diverse_instruction(temp_word, target_words_list)
            txt_filename = f"{idx:06d}_groundtruth.txt"
            txt_save_path = os.path.join(gt_dir, txt_filename)
            with open(txt_save_path, "w", encoding='utf-8') as f:
                f.write(instruction_text)

            # 6. 元数据记录
            final_metadata.append({
                "id": idx,
                "images": {
                    "input_bg": bg_path,               # 原始背景
                    "input_content": content_path,     # 文字排版参考
                    "mask": record['step1_item']['mask_path'],
                    "style_ref": record['style_path'], # 风格参考图
                    "groundtruth": gt_save_path
                },
                "instruction_file": txt_save_path,
                "text_info": {
                    "remove_word": temp_word,
                    "draw_word_str": target_word_str,
                    "style_desc": style_desc,
                    "generation_prompt": prompt_gen,    # 记录下生成 GT 用的 prompt
                    "full_instruction": instruction_text
                }
            })
            
        except Exception as e:
            print(f" -> Error processing GT for {idx}: {e}")
            continue

    # 保存最终 JSON
    json_name = f"final_training_dataset_rank{args.rank}.json"
    final_json_path = os.path.join(config.OUTPUT_DIR, json_name)
    
    with open(final_json_path, "w", encoding='utf-8') as f:
        json.dump(final_metadata, f, indent=4, ensure_ascii=False)
        
    print(f"\n=== All Done (Rank {args.rank})! ===")
    print(f"Output JSON: {final_json_path}")

if __name__ == "__main__":
    process_step3_and_4()