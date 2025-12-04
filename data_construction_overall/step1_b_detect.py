import os
import json
import cv2
import glob
import argparse
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

import config

def detect_and_generate_masks():
    # === 0. 参数解析 ===
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0, help="当前进程编号")
    parser.add_argument("--world_size", type=int, default=1, help="总进程数")
    args = parser.parse_args()

    print(f"--- Step 1-B: Detecting Text & Generating Masks [Rank {args.rank}/{args.world_size}] ---")
    
    # 1. 读取 Step 1-A 生成的所有中间数据
    # 因为 Step 1-A 也是多进程的，会生成多个 json，这里我们需要读取所有，然后按本进程的 rank 过滤任务
    meta_dir = config.SUB_DIRS["meta"]
    json_pattern = os.path.join(meta_dir, "step1_intermediate_rank*.json")
    json_files = glob.glob(json_pattern)
    
    items = []
    if not json_files:
        # 兼容旧版本单文件情况
        single_file = os.path.join(meta_dir, "step1_intermediate.json")
        if os.path.exists(single_file):
            with open(single_file, "r", encoding='utf-8') as f:
                items = json.load(f)
        else:
            print(f"Error: No intermediate files found in {meta_dir}")
            print("Please run step1_a_edit.py first.")
            return
    else:
        # 合并所有中间文件
        print(f"Loading intermediate metadata from {len(json_files)} files...")
        for jf in json_files:
            try:
                with open(jf, "r", encoding='utf-8') as f:
                    data = json.load(f)
                    items.extend(data)
            except Exception as e:
                print(f"Warning: Failed to load {jf}: {e}")

    # 2. 初始化 OCR
    # 仅当有需要检测的任务时才初始化，但在循环外初始化比较好
    # PaddleOCR 默认会尝试使用 GPU，如果多卡运行请通过 CUDA_VISIBLE_DEVICES 控制，或者在 paddleocr 初始化时指定 device_id
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        ocr_version="PP-OCRv5",
        # device="gpu", # 建议由外部环境变量 CUDA_VISIBLE_DEVICES 控制
        text_det_limit_type="max",
        text_det_limit_side_len=4000,
        text_det_thresh=0.5,
        text_det_box_thresh=0.5,
        text_det_unclip_ratio=1.5,
        text_rec_score_thresh=0.3
    )
    
    final_metadata = []
    
    print(f"Total items found: {len(items)}. Processing subset for Rank {args.rank}...")

    for item in items:
        idx = item['id']
        
        # === 分布式过滤：只处理属于当前 rank 的数据 ===
        if idx % args.world_size != args.rank:
            continue

        edited_path = item['edited_temp_path']
        original_bg_path = item['original_bg_path']
        temp_word = item.get('temp_word_used', "")
        
        # 检查是否是“不加字”的样本
        is_empty_sample = (temp_word == "")

        print(f"[{idx}] Processing...", end="")
        
        try:
            # 检查文件
            if not os.path.exists(edited_path):
                print(" -> File missing, skipping.")
                continue

            # 创建 Mask 画布 (全黑)
            mask = np.zeros((config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]), dtype=np.uint8)
            
            # === 分支逻辑 ===
            if is_empty_sample:
                # Case A: 本来就没加字 -> 不需要 OCR -> Mask 全黑
                print(" (No Text Sample) -> Zero Mask Generated.", end="")
                # Mask 保持全黑即可
            else:
                # Case B: 加了字 -> 运行 OCR -> 生成 Mask
                edited_img = Image.open(edited_path).convert("RGB")
                edited_np = np.array(edited_img)
                
                result = ocr.predict(edited_np)
                
                found = False
                if result and len(result) > 0:
                    page_result = result[0]
                    if isinstance(page_result, dict):
                        dt_polys = page_result.get("dt_polys", [])
                        rec_scores = page_result.get("rec_scores", [])
                        
                        if len(dt_polys) > 0 and len(rec_scores) > 0:
                            for i, score in enumerate(rec_scores):
                                if score > 0.8:
                                    if i < len(dt_polys):
                                        box = dt_polys[i]
                                        points = np.array(box).astype(np.int32)
                                        cv2.fillPoly(mask, [points], 255)
                                        found = True
                
                if not found:
                    print(" -> No text detected by OCR (might be obscured).", end="")
                    # 即使没检测到，也保留样本，Mask 为黑，意味着“无编辑区域”
                else:
                    print(" -> Mask Generated.", end="")

            # === 保存最终文件 ===
            # 1. 保存 Mask
            mask_name = f"{idx:06d}_mask.png"
            save_mask_path = os.path.join(config.SUB_DIRS["mask"], mask_name)
            Image.fromarray(mask).save(save_mask_path)
            
            # 2. 保存 原图 (Input Image)
            # 必须重新读取原始背景并 resize，确保与 mask 尺寸一致
            clean_input_img = Image.open(original_bg_path).convert("RGB").resize(config.IMAGE_SIZE)
            input_name = f"{idx:06d}_input.jpg"
            save_input_path = os.path.join(config.SUB_DIRS["image"], input_name)
            clean_input_img.save(save_input_path, quality=95)
            
            # 3. 记录元数据
            final_metadata.append({
                "id": idx,
                "input_path": save_input_path,
                "mask_path": save_mask_path,
                "original_bg": item['bg_name'],
                "temp_word_used": temp_word
            })
            print(" -> Done.")

        except Exception as e:
            print(f" -> Error: {e}")
            continue

    # 保存 Step 1 最终 Json (带 rank 后缀)
    final_json_name = f"step1_rank{args.rank}.json"
    final_json_path = os.path.join(config.SUB_DIRS["meta"], final_json_name)
    
    with open(final_json_path, "w", encoding='utf-8') as f:
        json.dump(final_metadata, f, indent=4, ensure_ascii=False)
        
    print(f"\n--- Step 1-B (Rank {args.rank}) Complete ---")
    print(f"Processed: {len(final_metadata)} samples")
    print(f"Saved to: {final_json_path}")

if __name__ == "__main__":
    detect_and_generate_masks()