import os
import json
import random
import sys
import importlib.util
import argparse
import glob
import numpy as np
import torch
import cv2
import onnxruntime as ort
from PIL import Image
from diffusers import DiffusionPipeline, OvisImagePipeline

import config

# ==========================================
# 1. 核心组件：多样化指令生成器 (适配列表)
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

# ==========================================
# 2. 分割模型类
# ==========================================
def refine_foreground(fg_img: Image.Image, mask_img: Image.Image, r=2) -> Image.Image:
    fg_img = fg_img.convert("RGBA")
    mask_img = mask_img.convert("L")
    fg_img.putalpha(mask_img)
    return fg_img

class ImageSegmentation:
    def __init__(self, model_path, model_input_size=[1024, 1024]):
        self.model_input_size = model_input_size
        
        if not os.path.exists(model_path):
            fallback_path = "/app/cold1/code/texteditRoPE/bg_remove/briaai/RMBG-2.0/model.onnx"
            if os.path.exists(fallback_path):
                model_path = fallback_path
            else:
                raise FileNotFoundError(f"RMBG Model not found at: {model_path}")
            
        print(f"Loading Segmentation Model: {model_path}...")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            self.ort_session = ort.InferenceSession(model_path, providers=providers)
        except Exception as e:
            print(f"Warning: Failed to load with CUDA, falling back to CPU. Error: {e}")
            self.ort_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    def preprocess_image(self, im: np.ndarray) -> np.ndarray:
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        im_resized = np.array(Image.fromarray(im).resize(self.model_input_size, Image.BILINEAR))
        image = im_resized.astype(np.float32) / 255.0
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        image = (image - mean) / std
        image = image.transpose(2, 0, 1)
        return np.expand_dims(image, axis=0)

    def postprocess_image(self, result: np.ndarray, im_size: list) -> np.ndarray:
        result = np.squeeze(result)
        result = np.array(Image.fromarray(result).resize(im_size, Image.BILINEAR))
        ma, mi = result.max(), result.min()
        result = (result - mi) / (ma - mi)
        return (result * 255).astype(np.uint8)

    def segment_image(self, img_obj) -> Image.Image:
        if isinstance(img_obj, str):
            orig_image = Image.open(img_obj).convert("RGB")
        elif isinstance(img_obj, Image.Image):
            orig_image = img_obj.convert("RGB")
        else:
            raise ValueError("Input must be path or PIL Image")

        image_size = orig_image.size
        image_array = np.array(orig_image)
        
        preprocessed = self.preprocess_image(image_array)
        ort_inputs = {self.ort_session.get_inputs()[0].name: preprocessed}
        ort_outs = self.ort_session.run(None, ort_inputs)
        result = ort_outs[0]
        
        mask_array = self.postprocess_image(result[0][0], image_size)
        pil_mask = Image.fromarray(mask_array).convert("L")
        
        no_bg_image = refine_foreground(orig_image, pil_mask)
        return no_bg_image

def load_word_list_from_module(path):
    spec = importlib.util.spec_from_file_location("word_list_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WORD_LIST

# ==========================================
# 3. 主逻辑
# ==========================================
def process_step3_and_4():
    ort.set_default_logger_severity(3)
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0, help="Current process rank")
    parser.add_argument("--world_size", type=int, default=1, help="Total number of processes")
    args = parser.parse_args()

    print(f"=== Starting Combined Step 3 (Style) & Step 4 (GT) Generation [Rank {args.rank}/{args.world_size}] ===")
    
    if not os.path.exists(config.STYLE_FILE):
        print(f"Error: Style file not found at {config.STYLE_FILE}")
        return
    with open(config.STYLE_FILE, 'r', encoding='utf-8') as f:
        styles_corpus = json.load(f)

    word_list = load_word_list_from_module(config.WORD_LIST)

    # 读取 Step 1 和 Step 2
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
    # PHASE 1: Generate Styles (Step 3)
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
            
        print(f"[{idx}] Generating Style: '{style_ref_word}'...")
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
    print(f"\n--- Phase 2: Generating GT & Diverse Instructions (Rank {args.rank}) ---")

    rmbg_path = getattr(config, "RMBG_MODEL_PATH", "/app/cold1/code/texteditRoPE/bg_remove/briaai/RMBG-2.0/model.onnx")
    segmenter = ImageSegmentation(model_path=rmbg_path)

    print(f"Loading Edit Model: {config.MODEL_QWEN_PATH}")
    pipe_edit = DiffusionPipeline.from_pretrained(
        config.MODEL_QWEN_PATH, 
        torch_dtype=torch.bfloat16, 
        device_map="balanced" 
    )

    final_metadata = []
    gt_dir = os.path.join(config.OUTPUT_DIR, "groundtruth")
    os.makedirs(gt_dir, exist_ok=True)

    # === [调试新增 1] 创建调试目录 ===
    debug_raw_dir = os.path.join(config.OUTPUT_DIR, "debug_raw_gen")
    debug_seg_dir = os.path.join(config.OUTPUT_DIR, "debug_segmented")
    os.makedirs(debug_raw_dir, exist_ok=True)
    os.makedirs(debug_seg_dir, exist_ok=True)
    # ===============================

    for record in processed_records:
        idx = record['id']
        temp_word = record['step1_item'].get('temp_word_used', "")
        target_word_str = record['step2_item']['target_word'] 
        target_words_list = record['step2_item'].get('target_words_list', [target_word_str])
        
        content_path = record['step2_item']['content_path']
        style_desc = record['assigned_style']
        bg_path = record['step1_item']['input_path'] 
        
        print(f"[{idx}] GT: Remove '{temp_word}' -> Draw {target_words_list}")

        try:
            # A. Qwen Edit 生成白底字
            content_img = Image.open(content_path).convert("RGB")
            prompt_edit = f"用{style_desc}的风格书写图中的文字“{target_word_str}”，让文字样式符合描述的风格，保持布局不变，背景保持使用纯白色,文字内容不变。"
            
            gen = torch.Generator(device="cuda").manual_seed(1234)
            i1_image = pipe_edit(
                prompt=prompt_edit,
                image=content_img,
                num_inference_steps=30,
                generator=gen
            ).images[0]

            # === [调试新增 2] 保存生成模型输出的原始图 (白底) ===
            raw_save_path = os.path.join(debug_raw_dir, f"{idx:06d}_raw.jpg")
            i1_image.save(raw_save_path)
            # ===============================================

            # B. 分割
            fg_transparent = segmenter.segment_image(i1_image)
            
            # === [调试新增 3] 保存分割后的透明图 (.png) ===
            seg_save_path = os.path.join(debug_seg_dir, f"{idx:06d}_seg.png")
            fg_transparent.save(seg_save_path)
            # ===========================================

            # C. 贴回
            background = Image.open(bg_path).convert("RGBA").resize(config.IMAGE_SIZE)
            if fg_transparent.size != background.size:
                fg_transparent = fg_transparent.resize(background.size, Image.BILINEAR)
            
            final_composition = Image.alpha_composite(background, fg_transparent)
            final_gt = final_composition.convert("RGB")
            
            # D. 保存 GT
            gt_filename = f"{idx:06d}_groundtruth.jpg"
            gt_save_path = os.path.join(gt_dir, gt_filename)
            final_gt.save(gt_save_path, quality=95)
            
            # E. 生成多样化指令
            instruction_text = generate_diverse_instruction(temp_word, target_words_list)
            txt_filename = f"{idx:06d}_groundtruth.txt"
            txt_save_path = os.path.join(gt_dir, txt_filename)
            with open(txt_save_path, "w", encoding='utf-8') as f:
                f.write(instruction_text)

            # F. 元数据
            final_metadata.append({
                "id": idx,
                "images": {
                    "input_original": bg_path,
                    "mask": record['step1_item']['mask_path'],
                    "content": content_path,
                    "style": record['style_path'],
                    "groundtruth": gt_save_path,
                    # 可以选择把调试路径也记录进去，方便查看json直接定位
                    "debug_raw": raw_save_path, 
                    "debug_seg": seg_save_path
                },
                "instruction_file": txt_save_path,
                "text_info": {
                    "remove_word": temp_word,
                    "draw_word_str": target_word_str,
                    "draw_words_list": target_words_list,
                    "style_desc": style_desc,
                    "full_instruction": instruction_text
                }
            })
            
        except Exception as e:
            print(f" -> Error processing GT for {idx}: {e}")
            continue

    json_name = f"final_training_dataset_rank{args.rank}.json"
    final_json_path = os.path.join(config.OUTPUT_DIR, json_name)
    
    with open(final_json_path, "w", encoding='utf-8') as f:
        json.dump(final_metadata, f, indent=4, ensure_ascii=False)
        
    print(f"\n=== All Done (Rank {args.rank})! ===")
    print(f"Output JSON: {final_json_path}")

if __name__ == "__main__":
    process_step3_and_4()