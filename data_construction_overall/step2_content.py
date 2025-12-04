import os
import json
import random
import math
import importlib.util
import argparse
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config
from utils_bezier import bezier, bezier_tangent, get_text_width

# === 配置参数 ===
PROB_CURVED = 0.5     # 文字是弯曲风格的概率

def load_word_list_from_module(path):
    spec = importlib.util.spec_from_file_location("word_list_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WORD_LIST

def check_boundary(box, img_w, img_h):
    """
    检查 bbox 是否完全在图像范围内 (0, 0, img_w, img_h)
    box: (x1, y1, x2, y2)
    """
    if not box: return False
    x1, y1, x2, y2 = box
    # 严格模式，确保文字不被截断
    return x1 >= 0 and y1 >= 0 and x2 <= img_w and y2 <= img_h

def check_overlap(box1, box2):
    """
    简单的矩形碰撞检测 (x1, y1, x2, y2)
    虽然只生成一个词，但保留此函数以便未来扩展或与既有逻辑兼容
    """
    if not box1 or not box2: return False
    return not (box1[0] > box2[2] or box1[2] < box2[0] or box1[1] > box2[3] or box1[3] < box2[1])

def draw_normal_text(temp_layer, text, font, tracking=0):
    """
    绘制正常风格（可旋转矩形）的文字
    """
    total_w, char_widths = get_text_width(text, font, tracking)
    if total_w == 0: return None
    
    ascent, descent = font.getmetrics()
    text_h = ascent + descent
    
    # 创建小画布
    temp_w = int(total_w * 1.2)
    temp_h = int(text_h * 1.5)
    txt_img = Image.new('RGBA', (temp_w, temp_h), (255, 255, 255, 0))
    d = ImageDraw.Draw(txt_img)
    
    d.text(((temp_w - total_w) // 2, (temp_h - text_h) // 2), text, font=font, fill="black")
    
    # 随机旋转
    angle = random.uniform(-45, 45)
    rotated_txt = txt_img.rotate(angle, resample=Image.BICUBIC, expand=True)
    
    w_final, h_final = rotated_txt.size
    img_w, img_h = temp_layer.size
    
    # 边界保护 (Margin)
    margin = 20
    max_x = img_w - w_final - margin
    max_y = img_h - h_final - margin
    
    # 如果图片太小放不下文字，直接返回 None
    if max_x < margin or max_y < margin: return None
    
    paste_x = random.randint(margin, max_x)
    paste_y = random.randint(margin, max_y)
    
    temp_layer.paste(rotated_txt, (paste_x, paste_y), mask=rotated_txt)
    
    # 返回 bbox (x1, y1, x2, y2)
    return (paste_x, paste_y, paste_x + w_final, paste_y + h_final)

def draw_curved_text(temp_layer, text, font, tracking=0, curve_intensity=0.5):
    """
    绘制弯曲风格的文字
    """
    total_w, char_widths = get_text_width(text, font, tracking)
    if total_w == 0: return None
    
    img_w, img_h = temp_layer.size
    
    margin = 50
    # 起点限制
    max_start_x = max(margin, img_w - int(total_w) - margin)
    start_x = random.randint(margin, max_start_x)
    start_y = random.randint(img_h // 4, img_h // 4 * 3)
    
    arch_h = total_w * curve_intensity * random.choice([1, -1]) 
    
    p0 = (start_x, start_y)
    p3 = (start_x + total_w, start_y)
    p1 = (start_x + total_w * random.uniform(0.2, 0.4), start_y - arch_h)
    p2 = (start_x + total_w * random.uniform(0.6, 0.8), start_y - arch_h)
    
    current_dist = 0
    
    # 实时追踪真实坐标
    real_min_x, real_min_y = float('inf'), float('inf')
    real_max_x, real_max_y = float('-inf'), float('-inf')
    
    drawn_something = False

    for i, ch in enumerate(text):
        w = char_widths[i]
        center_pos = current_dist + (w / 2)
        t = center_pos / max(1, total_w)
        t = max(0, min(1, t))
        
        pt = bezier(t, p0, p1, p2, p3)
        tg = bezier_tangent(t, p0, p1, p2, p3)
        angle = math.degrees(math.atan2(tg[1], tg[0]))
        
        char_size = int(font.size * 2.5)
        char_img = Image.new("RGBA", (char_size, char_size), (255,255,255,0))
        d = ImageDraw.Draw(char_img)
        bbox = font.getbbox(ch)
        if not bbox: continue
        
        text_h_val = bbox[3] - bbox[1]
        d.text((char_size//2 - w/2, char_size//2 - text_h_val/2 - bbox[1]), ch, font=font, fill="black")
        
        rotated = char_img.rotate(-angle, resample=Image.BICUBIC, expand=True)
        
        # 计算粘贴坐标
        paste_x = int(pt[0] - rotated.width / 2)
        paste_y = int(pt[1] - rotated.height / 2)
        
        # 粘贴
        temp_layer.paste(rotated, (paste_x, paste_y), mask=rotated)
        
        # 更新真实 Bounding Box
        real_min_x = min(real_min_x, paste_x)
        real_min_y = min(real_min_y, paste_y)
        real_max_x = max(real_max_x, paste_x + rotated.width)
        real_max_y = max(real_max_y, paste_y + rotated.height)
        
        drawn_something = True
        current_dist += w + tracking

    if not drawn_something:
        return None

    return (int(real_min_x), int(real_min_y), int(real_max_x), int(real_max_y))

def generate_contents():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    args = parser.parse_args()

    print(f"--- Step 2: Generating Content [Rank {args.rank}] ---")
    
    word_list = load_word_list_from_module(config.WORD_LIST)
    
    # 读取 step1 数据
    meta_dir = config.SUB_DIRS["meta"]
    json_pattern = os.path.join(meta_dir, "step1_rank*.json")
    json_files = glob.glob(json_pattern)
    step1_data = []
    
    if not json_files:
        # Fallback to single file if rank files not found
        single_file = os.path.join(meta_dir, "step1.json")
        if os.path.exists(single_file):
             with open(single_file, "r", encoding='utf-8') as f:
                step1_data = json.load(f)
    else:
        for jf in json_files:
            with open(jf, "r", encoding='utf-8') as f:
                step1_data.extend(json.load(f))
            
    metadata = []
    
    for item in step1_data:
        idx = item['id']
        if idx % args.world_size != args.rank: continue
        
        # === 修改处：强制只生成 1 个词 ===
        num_texts = 1 
        # ===============================
            
        bg = Image.new("RGB", config.IMAGE_SIZE, (255,255,255))
        img_w, img_h = config.IMAGE_SIZE
        
        generated_words = []
        occupied_boxes = [] 
        
        # 这里的循环实际上只会执行一次 (range(1))
        for _ in range(num_texts):
            target_text = random.choice(word_list)
            font_size = random.randint(100, 180)
            tracking = random.randint(0, 8)
            
            try:
                font = ImageFont.truetype(config.FONT_PATH, font_size)
            except:
                print("Font Error")
                return

            is_curved = random.random() < PROB_CURVED
            
            # 尝试多次放置
            for attempt in range(15): # 增加尝试次数，确保尽量成功
                temp_layer = Image.new("RGBA", config.IMAGE_SIZE, (0,0,0,0))
                
                bbox = None
                if is_curved:
                    curve_intensity = random.uniform(0.1, 0.6)
                    bbox = draw_curved_text(temp_layer, target_text, font, tracking, curve_intensity)
                else:
                    bbox = draw_normal_text(temp_layer, target_text, font, tracking)
                
                if bbox:
                    # 1. 检查边界
                    if not check_boundary(bbox, img_w, img_h):
                        continue

                    # 2. 检查重叠 (虽然现在只有1个词，但保留逻辑无害且通用)
                    overlap = False
                    for exist_box in occupied_boxes:
                        if check_overlap(bbox, exist_box):
                            overlap = True
                            break
                    
                    if not overlap:
                        # 成功：合成
                        bg.paste(temp_layer, (0,0), mask=temp_layer)
                        occupied_boxes.append(bbox)
                        generated_words.append(target_text)
                        break 
                    
        if generated_words:
            save_name = f"{idx:06d}_content.jpg"
            save_path = os.path.join(config.SUB_DIRS["content"], save_name)
            bg.save(save_path)
            
            metadata.append({
                "id": idx,
                "content_path": save_path,
                "target_word": generated_words[0], # 直接取第一个词
                "target_words_list": generated_words
            })
            print(f"[{idx}] Generated: {generated_words}")
        else:
            print(f"[{idx}] Failed to generate text.")

    json_name = f"step2_rank{args.rank}.json"
    output_json_path = os.path.join(config.SUB_DIRS["meta"], json_name)
    with open(output_json_path, "w", encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    print("Done.")

if __name__ == "__main__":
    generate_contents()