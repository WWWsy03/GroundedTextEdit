import os
import json
import math
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# 1. 基础数学工具 (保持不变)
# ==========================================
def bezier_point(t, p0, p1, p2, p3):
    """ 计算贝塞尔曲线上的点坐标 (x, y) """
    x = p0[0]*(1-t)**3 + 3*p1[0]*t*(1-t)**2 + 3*p2[0]*t**2*(1-t) + p3[0]*t**3
    y = p0[1]*(1-t)**3 + 3*p1[1]*t*(1-t)**2 + 3*p2[1]*t**2*(1-t) + p3[1]*t**3
    return x, y

def bezier_tangent(t, p0, p1, p2, p3):
    """ 计算贝塞尔曲线上的切线向量 (dx, dy) """
    dx = -3*p0[0]*(1-t)**2 + 3*p1[0]*(1-t)**2 - 6*p1[0]*t*(1-t) + \
         6*p2[0]*t*(1-t) - 3*p2[0]*t**2 + 3*p3[0]*t**2
    dy = -3*p0[1]*(1-t)**2 + 3*p1[1]*(1-t)**2 - 6*p1[1]*t*(1-t) + \
         6*p2[1]*t*(1-t) - 3*p2[1]*t**2 + 3*p3[1]*t**2
    return dx, dy

def get_text_width(text, font, tracking=0):
    """ 计算包含字间距的文字总宽度 """
    if not text: return 0, []
    char_widths = [font.getlength(ch) for ch in text]
    total_width = sum(char_widths) + tracking * (len(text) - 1)
    return total_width, char_widths

# ==========================================
# 2. 核心绘制逻辑 (保持不变)
# ==========================================
def draw_curved_text_core(
    base_img, 
    text, 
    font, 
    center_x, 
    center_y, 
    curve_intensity=0.5, 
    direction='up', 
    tracking=0,
    debug=True
):
    draw_debug = ImageDraw.Draw(base_img)
    total_w, char_widths = get_text_width(text, font, tracking)
    if total_w == 0: return
    
    # 定义贝塞尔曲线控制点
    start_x = center_x - total_w / 2
    end_x = center_x + total_w / 2
    base_y = center_y 
    
    arch_h = total_w * curve_intensity
    if direction == 'down':
        arch_h = -arch_h 
        
    p0 = (start_x, base_y)
    p3 = (end_x, base_y)
    p1 = (start_x + total_w * 0.33, base_y - arch_h)
    p2 = (start_x + total_w * 0.66, base_y - arch_h)

    # Debug 辅助线
    if debug:
        draw_debug.line([p0, p1, p2, p3], fill="lightgray", width=2)
        debug_points = [bezier_point(t, p0, p1, p2, p3) for t in np.linspace(0, 1, 50)]
        draw_debug.line(debug_points, fill="red", width=3)

    current_dist = 0
    real_min_x, real_min_y = float('inf'), float('inf')
    real_max_x, real_max_y = float('-inf'), float('-inf')

    for i, ch in enumerate(text):
        w = char_widths[i]
        center_pos = current_dist + (w / 2)
        t = center_pos / max(1, total_w)
        t = max(0, min(1, t))
        
        pt = bezier_point(t, p0, p1, p2, p3)
        tg = bezier_tangent(t, p0, p1, p2, p3)
        angle = math.degrees(math.atan2(tg[1], tg[0]))
        
        char_size = int(font.size * 2.0)
        char_img = Image.new("RGBA", (char_size, char_size), (255,255,255,0))
        d_char = ImageDraw.Draw(char_img)
        bbox = font.getbbox(ch)
        
        if bbox: 
            char_h = bbox[3] - bbox[1]
            draw_x = char_size//2 - w/2
            draw_y = char_size//2 - char_h/2 - bbox[1]
            d_char.text((draw_x, draw_y), ch, font=font, fill="black")
            
            rotated = char_img.rotate(-angle, resample=Image.BICUBIC, expand=True)
            paste_x = int(pt[0] - rotated.width / 2)
            paste_y = int(pt[1] - rotated.height / 2)
            
            base_img.paste(rotated, (paste_x, paste_y), mask=rotated)
            
            real_min_x = min(real_min_x, paste_x)
            real_min_y = min(real_min_y, paste_y)
            real_max_x = max(real_max_x, paste_x + rotated.width)
            real_max_y = max(real_max_y, paste_y + rotated.height)
        
        current_dist += w + tracking

    if debug and real_min_x != float('inf'):
        draw_debug.rectangle((real_min_x, real_min_y, real_max_x, real_max_y), outline="blue", width=2)

# ==========================================
# 3. 主程序入口 (修改为读取 JSON)
# ==========================================
if __name__ == "__main__":

    output_path = "/app/cold1/code/texteditRoPE/data_construction_overall/test_content/test.png"


    # 1. 读取配置文件
    default_config = {
"text": "helloworld",
"font_path": "arial.ttf",
"font_size": 80,
"curve_intensity": 0.3,
"curve_direction": "up",
"tracking": 5,
"canvas_size": [1024, 1024],
"output_path": "/app/cold1/code/texteditRoPE/data_construction_overall/test_content/test.png",
}


    # 2. 准备参数
    text = default_config.get("text", "Default Text")
    font_path = default_config.get("font_path", "arial.ttf")
    font_size = default_config.get("font_size", 100)
    curve_intensity = default_config.get("curve_intensity", 0.5)
    direction = default_config.get("curve_direction", "up")
    tracking = default_config.get("tracking", 0)
    w, h = default_config.get("canvas_size", [1024, 1024])
    

    # 3. 准备画布
    bg = Image.new("RGB", (w, h), (255, 255, 255))

    # 4. 加载字体
    try:
        font = ImageFont.truetype(font_path, font_size)
    except OSError:
        print(f"Warning: Font '{font_path}' not found. Trying default system font...")
        try:
            # 这里的路径可能需要根据你的系统修改
            font = ImageFont.truetype("/app/cold1/simhei.ttf", font_size)
        except:
            print("Error: Could not load any font. Check 'font_path' in json.")
            exit()

    # 5. 计算绘制中心
    cx, cy = w // 2, h // 2
    # 为了视觉平衡，根据拱形方向调整中心Y轴
    if direction == 'up':
        cy = cy + int(h * 0.1)
    else:
        cy = cy - int(h * 0.1)

    # 6. 执行绘制
    draw_curved_text_core(
        bg, 
        text, 
        font, 
        center_x=cx, 
        center_y=cy,
        curve_intensity=curve_intensity,
        direction=direction,
        tracking=tracking,
    )

    # 7. 保存结果
    bg.save(output_path)
    print(f"Successfully saved to: {os.path.abspath(output_path)}")