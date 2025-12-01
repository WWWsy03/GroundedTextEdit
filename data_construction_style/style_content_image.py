import glob
import os
import json
import math
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR
from assets.word_list import INSTRUCTION_TEMPLATES

# === 配置 ===
# 指向包含那8个 part json 的文件夹路径
STEP2_METADATA_DIR = "/app/cold1/code/texteditRoPE/data_construction_style"
# 文件名前缀匹配模式
STEP2_FILE_PATTERN = "dataset_metadata_step2_final_part*.json"

OUTPUT_INPUT_IMG_DIR = "/app/cold1/code/texteditRoPE/data_construction_style/dataset_images_3/content_images"
# 输出为一个总的 JSON
FINAL_DATASET_JSON = "final_training_dataset_3.json"
REJECTED_DATASET_JSON = "rejected_samples_3.json"

FONT_PATH = "/app/cold1/simhei.ttf" 
CANVAS_SIZE = (1024, 1024)
CONFIDENCE_THRESHOLD = 0.8

os.makedirs(OUTPUT_INPUT_IMG_DIR, exist_ok=True)

os.makedirs(OUTPUT_INPUT_IMG_DIR, exist_ok=True)

# === 1. 初始化 PaddleOCR (参考你提供的配置) ===
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    ocr_version="PP-OCRv5",
    # device="gpu", # 如果没有GPU请注释掉或改为cpu
    text_det_limit_type="max",
    text_det_limit_side_len=4000,
    text_det_thresh=0.5, # 稍微放宽阈值以检测更多文本
    text_det_box_thresh=0.5,
    text_det_unclip_ratio=1.5,
    text_rec_score_thresh=0.3,
)

# === 2. 辅助工具函数 ===

def get_rotated_box_geometry(box):
    """
    根据 OCR 返回的四个点计算中心点、宽高和旋转角度
    box: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    """
    # 关键：确保输入是 float32 类型的 numpy 数组，避免 np.integer 报错
    box = np.array(box, dtype=np.float32)
    
    # 1. 计算中心点
    center_x = float(np.mean(box[:, 0]))
    center_y = float(np.mean(box[:, 1]))
    
    # 2. 计算宽度 (取上下两边的平均长度)
    w1 = np.linalg.norm(box[0] - box[1])
    w2 = np.linalg.norm(box[3] - box[2])
    width = float((w1 + w2) / 2)

    # 3. 计算高度 (取左右两边的平均长度)
    h1 = np.linalg.norm(box[0] - box[3])
    h2 = np.linalg.norm(box[1] - box[2])
    height = float((h1 + h2) / 2)

    # 4. 计算角度
    # 使用 top edge (p0 -> p1) 计算角度
    dx = box[1][0] - box[0][0]
    dy = box[1][1] - box[0][1]
    
    # atan2(y, x)，图像坐标系 y 向下
    angle_rad = math.atan2(dy, dx) 
    angle_deg = math.degrees(angle_rad)

    return (center_x, center_y), width, height, angle_deg

def create_rotated_text_image(text, target_w, target_h, angle_deg, font_path):
    """
    生成一张包含旋转文字的透明底图，文字大小自动适应 target_w/h
    """
    font_size = 10
    try:
        font = ImageFont.truetype(font_path, font_size)
    except OSError:
        print(f"  -> Error: Font file not found: {font_path}")
        return None

    # 稍微留一点边距
    safe_w = target_w * 0.98
    safe_h = target_h * 0.98

    # 1. 寻找最大可用字号
    while True:
        bbox = font.getbbox(text) # (left, top, right, bottom)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        if text_w > safe_w or text_h > safe_h:
            font_size = max(10, font_size - 1)
            break
        font_size += 2 
        font = ImageFont.truetype(font_path, font_size)
    
    # 最终字体
    font = ImageFont.truetype(font_path, font_size)
    bbox = font.getbbox(text)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # 2. 创建足够大的画布以支持旋转
    temp_size = int(max(target_w, target_h) * 2.5) + 100
    txt_img = Image.new('RGBA', (temp_size, temp_size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt_img)

    # 3. 居中绘制 (在临时画布中心)
    # 计算文字左上角坐标，使得文字中心对齐画布中心
    text_x = (temp_size - text_w) / 2 - bbox[0]
    text_y = (temp_size - text_h) / 2 - bbox[1]
    
    draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0, 255))

    # 4. 旋转 (PIL rotate 是逆时针，所以取负角度)
    # 使用 BICUBIC 保证边缘平滑
    rotated_txt_img = txt_img.rotate(-angle_deg, resample=Image.BICUBIC, expand=False)

    return rotated_txt_img

def normalize_text(text):
    """归一化文本：去除空格、转小写"""
    if not isinstance(text, str): return ""
    return text.replace(" ", "").lower()
    
# === 3. 主处理逻辑 ===
def process_ocr_and_generate_input():
    print("--- Step 3: OCR Detection, Filtering & Rendering (Batch Merge) ---")
    
    if not os.path.exists(FONT_PATH):
        print(f"Fatal Error: Font not found at {FONT_PATH}")
        return

    # --- 修改部分：自动加载所有 part json 文件 ---
    all_metadata_step2 = []
    
    # 构造 glob 搜索路径
    search_path = os.path.join(STEP2_METADATA_DIR, STEP2_FILE_PATTERN)
    json_files = sorted(glob.glob(search_path))
    
    if not json_files:
        print(f"Error: No JSON files found matching {search_path}")
        return
        
    print(f"Found {len(json_files)} metadata files. Loading...")
    
    for jf in json_files:
        print(f"  -> Loading {os.path.basename(jf)}...")
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_metadata_step2.extend(data)
        except Exception as e:
            print(f"Warning: Failed to load {jf}: {e}")

    print(f"Total items loaded: {len(all_metadata_step2)}")
    # -----------------------------------------------

    final_dataset = []
    rejected_dataset = [] 

    total_items = len(all_metadata_step2)

    for idx, item in enumerate(all_metadata_step2):
        target_path = item["target"]["image_path"]
        target_word = item["target"]["word"] 
        
        print(f"[{idx}/{total_items}] ID:{item['pair_id']} Word:'{target_word}'...", end="")

        try:
            result = ocr.predict(target_path)
        except Exception as e:
            print(f" -> Error: OCR Failed {e}")
            continue

        input_img = Image.new('RGB', CANVAS_SIZE, (255, 255, 255))
        
        valid_boxes_count = 0
        collected_boxes = [] 
        collected_texts = [] 

        if result and len(result) > 0:
            page_result = result[0]
            dt_polys = page_result.get("dt_polys", [])
            rec_scores = page_result.get("rec_scores", [])
            rec_texts = page_result.get("rec_texts", [])
            
            if len(dt_polys) > 0 and len(rec_scores) > 0:
                for i, score in enumerate(rec_scores):
                    # 1. 置信度过滤
                    if score < CONFIDENCE_THRESHOLD:
                        continue
                        
                    current_box = dt_polys[i]
                    current_text_segment = rec_texts[i] 

                    # 2. 渲染逻辑
                    try:
                        center, w, h, angle = get_rotated_box_geometry(current_box)
                        rotated_layer = create_rotated_text_image(current_text_segment, w, h, angle, FONT_PATH)
                        
                        if rotated_layer is None: continue

                        layer_w, layer_h = rotated_layer.size
                        paste_x = int(center[0] - layer_w / 2)
                        paste_y = int(center[1] - layer_h / 2)
                        
                        input_img.paste(rotated_layer, (paste_x, paste_y), mask=rotated_layer)
                        
                        # 收集数据
                        valid_boxes_count += 1
                        box_list = current_box.tolist() if isinstance(current_box, np.ndarray) else current_box
                        collected_boxes.append(box_list)
                        collected_texts.append(current_text_segment)
                        
                    except Exception as e:
                        print(f" (Render Error: {e})", end="")
                        continue
        
        # === 一致性校验 ===
        if valid_boxes_count > 0:
            detected_full_string = "".join(collected_texts)
            norm_detected = normalize_text(detected_full_string)
            norm_target = normalize_text(target_word)
            
            if norm_detected == norm_target:
                # --- 保存 ---
                # 注意：为了防止不同 part 文件的 id 冲突（如果有），这里最好保留原始文件名逻辑
                # 或者直接用全局 idx，但为了和图片对应，使用 item['pair_id'] 最稳
                # 假设 Step 2 中 pair_id 是全局唯一的 (0 到 Total-1)，这里就安全
                input_filename = f"pair_{item['pair_id']:05d}_input.jpg"
                input_full_path = os.path.join(OUTPUT_INPUT_IMG_DIR, input_filename)
                input_img.save(input_full_path, quality=95)
                
                template = random.choice(INSTRUCTION_TEMPLATES)
                instruction_text = template.format(word=target_word)

                final_entry = {
                    "id": item["pair_id"],
                    "input_image": input_full_path,
                    "style_image": item["reference"]["image_path"],
                    "target_image": target_path,
                    "instruction": instruction_text,
                    "metadata": {
                        "word": target_word,
                        "style_desc": item["style_description"],
                        "ocr_boxes": collected_boxes,
                        "ocr_texts": collected_texts,
                        "box_count": valid_boxes_count
                    }
                }
                final_dataset.append(final_entry)
                print(f" -> PASS")
            else:
                reject_info = {
                    "pair_id": item["pair_id"],
                    "expected": target_word,
                    "ocr_detected": collected_texts,
                    "ocr_joined": detected_full_string,
                    "reason": "Content Mismatch"
                }
                rejected_dataset.append(reject_info)
                print(f" -> REJECT (Content)")
        else:
            reject_info = {
                "pair_id": item["pair_id"],
                "expected": target_word,
                "ocr_detected": [],
                "reason": "Low Confidence/No Text"
            }
            rejected_dataset.append(reject_info)
            print(" -> REJECT (Empty)")

    # --- 保存最终合并结果 ---
    with open(FINAL_DATASET_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, indent=4, ensure_ascii=False)
        
    with open(REJECTED_DATASET_JSON, 'w', encoding='utf-8') as f:
        json.dump(rejected_dataset, f, indent=4, ensure_ascii=False)
        
    print(f"\n--- Step 3 Complete ---")
    print(f"Total Inputs: {total_items}")
    print(f"Valid Output: {len(final_dataset)} -> {FINAL_DATASET_JSON}")
    print(f"Rejected:     {len(rejected_dataset)} -> {REJECTED_DATASET_JSON}")
    
if __name__ == "__main__":
    process_ocr_and_generate_input()