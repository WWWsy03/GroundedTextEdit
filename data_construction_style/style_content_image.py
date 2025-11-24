import os
import json
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from paddleocr import PaddleOCR

# === 配置 ===
STEP2_METADATA = "/app/code/texteditRoPE/data_construction_style/dataset_metadata_step2_wip.json"
OUTPUT_INPUT_IMG_DIR = "dataset_images/content_images"
FINAL_DATASET_JSON = "final_training_dataset.json"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" # 请替换为你机器上的真实字体路径，如 msyh.ttc 或 arial.ttf
CANVAS_SIZE = (1024, 1024)

os.makedirs(OUTPUT_INPUT_IMG_DIR, exist_ok=True)

# 初始化 PaddleOCR (第一次运行会自动下载模型)
# use_angle_cls=True 对检测旋转文字很重要
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False) 

def get_rotated_box_geometry(box):
    """
    根据 OCR 返回的四个点计算中心点、宽高和旋转角度
    box: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] (通常顺序是 左上, 右上, 右下, 左下)
    """
    box = np.array(box, dtype=np.float32)
    
    # 1. 计算中心点
    center_x = np.mean(box[:, 0])
    center_y = np.mean(box[:, 1])
    
    # 2. 计算宽度 (取上下两边的平均长度)
    # top edge: p0 -> p1
    w1 = np.linalg.norm(box[0] - box[1])
    # bottom edge: p3 -> p2
    w2 = np.linalg.norm(box[3] - box[2])
    width = (w1 + w2) / 2

    # 3. 计算高度 (取左右两边的平均长度)
    # left edge: p0 -> p3
    h1 = np.linalg.norm(box[0] - box[3])
    # right edge: p1 -> p2
    h2 = np.linalg.norm(box[1] - box[2])
    height = (h1 + h2) / 2

    # 4. 计算角度 (相对于水平线的角度)
    # 使用 p0 和 p1 计算角度。注意图像坐标系 y 轴向下。
    dx = box[1][0] - box[0][0]
    dy = box[1][1] - box[0][1]
    # atan2 返回的是弧度，转换为角度
    # 图像坐标系中，如果 dy > 0 (右上点比左上点低)，说明顺时针旋转了
    angle_rad = math.atan2(dy, dx) 
    angle_deg = math.degrees(angle_rad)

    return (center_x, center_y), width, height, angle_deg

def create_rotated_text_image(text, target_w, target_h, angle_deg, font_path):
    """
    生成一张包含旋转文字的透明底图，文字大小自动适应 target_w/h
    """
    # 1. 寻找合适的字体大小
    font_size = 10
    # 步进式寻找最大可用字体
    # 优化：可以使用二分查找加快速度，这里用简单的迭代
    font = ImageFont.truetype(font_path, font_size)
    
    # 这里的 target_w/h 是旋转前的逻辑宽高
    # 我们给一点 padding 避免文字贴边太死
    safe_w = target_w * 0.95
    safe_h = target_h * 0.95

    while True:
        # getbbox returns (left, top, right, bottom)
        bbox = font.getbbox(text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        if text_w > safe_w or text_h > safe_h:
            font_size -= 1
            break
        font_size += 2 # 步长为2加快速度
        font = ImageFont.truetype(font_path, font_size)
    
    # 最终字体
    font = ImageFont.truetype(font_path, max(10, font_size))
    bbox = font.getbbox(text)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # 2. 创建临时画布绘制水平文字 (RGBA)
    # 画布尺寸要足够大以容纳旋转后的文字
    temp_canvas_size = int(max(target_w, target_h) * 2) + 100
    txt_img = Image.new('RGBA', (temp_canvas_size, temp_canvas_size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt_img)

    # 将文字绘制在临时画布正中心
    # xy 是文字左上角位置，计算居中位置
    text_x = (temp_canvas_size - text_w) / 2 - bbox[0]
    text_y = (temp_canvas_size - text_h) / 2 - bbox[1]
    
    draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0, 255)) # 纯黑字

    # 3. 旋转文字
    # Pillow 的 rotate 是逆时针的。
    # 如果 angle_deg 是根据 atan2(dy, dx) 算出来的，dy>0 代表顺时针倾斜（图像坐标系）。
    # 此时 angle_deg 是正数。要让 Pillow 顺时针转，需要负的角度。
    rotated_txt_img = txt_img.rotate(-angle_deg, resample=Image.BICUBIC, expand=False)

    return rotated_txt_img

def process_ocr_and_generate_input():
    print("--- Step 3: OCR and Input Image Generation ---")
    
    # 检查字体文件是否存在
    if not os.path.exists(FONT_PATH):
        # 尝试使用系统默认字体作为备选（仅针对Linux/Mac，Windows可能需要指定）
        print(f"Warning: Font not found at {FONT_PATH}. Trying default.")
    
    with open(STEP2_METADATA, 'r', encoding='utf-8') as f:
        metadata_step2 = json.load(f)

    final_dataset = []

    for idx, item in enumerate(metadata_step2):
        target_path = item["target"]["image_path"]
        target_word = item["target"]["word"]
        
        print(f"Processing {idx}: {target_word} | Path: {target_path}")

        # 1. 运行 OCR
        # ocr.ocr 返回结构: [ [ [box], [text, conf] ], ... ]
        # box 是 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        ocr_result = ocr.ocr(target_path, cls=True)

        best_box = None
        
        # 2. 筛选最佳匹配框
        # 策略：在 OCR 结果中找包含 target_word 的框。如果没有，找面积最大的框。
        found_match = False
        
        if ocr_result and ocr_result[0]:
            # 按面积排序，优先处理大字（防止识别到背景里的微小噪点）
            # 计算面积可以使用简单的 cv2.contourArea 或者 宽*高
            lines = ocr_result[0]
            
            # 尝试寻找包含目标文字的框 (不区分大小写)
            for line in lines:
                box = line[0]
                txt = line[1][0]
                if target_word.lower() in txt.lower() or txt.lower() in target_word.lower():
                    best_box = box
                    found_match = True
                    break
            
            # 如果没匹配到文字，取置信度最高的框作为 fallback
            if not best_box:
                # 按置信度排序
                lines.sort(key=lambda x: x[1][1], reverse=True)
                best_box = lines[0][0]
                print(f"  -> Warning: Word mismatch. Expected '{target_word}', OCR saw '{lines[0][1][0]}'. Using box anyway.")

        if best_box is None:
            print(f"  -> Error: OCR found nothing in {target_path}. Skipping.")
            continue

        # 3. 计算几何参数
        center, w, h, angle = get_rotated_box_geometry(best_box)
        
        # 4. 渲染输入图
        # 创建纯白背景
        input_img = Image.new('RGB', CANVAS_SIZE, (255, 255, 255))
        
        try:
            # 生成旋转后的文字图层
            rotated_text_layer = create_rotated_text_image(target_word, w, h, angle, FONT_PATH)
            
            # 计算粘贴位置 (将旋转图层的中心对齐到 OCR 框的中心)
            layer_w, layer_h = rotated_text_layer.size
            paste_x = int(center[0] - layer_w / 2)
            paste_y = int(center[1] - layer_h / 2)
            
            # 粘贴 (使用 alpha通道作为 mask)
            input_img.paste(rotated_text_layer, (paste_x, paste_y), mask=rotated_text_layer)
            
            # 保存
            input_filename = f"pair_{item['pair_id']:04d}_input.jpg"
            input_full_path = os.path.join(OUTPUT_INPUT_IMG_DIR, input_filename)
            input_img.save(input_full_path, quality=95)

            # 5. 构造最终数据
            # 构造指令，增加多样性
            # templates = [
            #     f'Change the text style of "{target_word}" to match the reference image.',
            #     f'Transfer the style from the reference to the text "{target_word}".',
            #     f'Apply the reference style to the word "{target_word}".',
            #     f'Make the text "{target_word}" look like the style in the reference image.'
            # ]
            instruction = f'Change the text style of "{target_word}" to match the reference image.'

            final_entry = {
                "id": item["pair_id"],
                "input_image": input_full_path,          # Source (白底黑字)
                "style_image": item["reference"]["image_path"], # Style Ref
                "target_image": target_path,             # GT
                "instruction": instruction,
                "metadata": {
                    "word": target_word,
                    "style_desc": item["style_description"],
                    "ocr_box": best_box,
                    "detected_angle": angle
                }
            }
            final_dataset.append(final_entry)
            
        except Exception as e:
            print(f"  -> Error during rendering: {e}")
            continue

    # 保存最终用于训练的 JSON
    with open(FINAL_DATASET_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, indent=4, ensure_ascii=False)
        
    print(f"--- Step 3 Done. Created {len(final_dataset)} triplets in {FINAL_DATASET_JSON} ---")

if __name__ == "__main__":
    process_ocr_and_generate_input()