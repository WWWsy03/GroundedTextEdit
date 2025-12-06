import os
import json
import random
import math
import glob
import argparse
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import DiffusionPipeline, OvisImagePipeline

# ==========================================
# 0. 基础工具: 贝塞尔曲线与文本计算
# ==========================================
def bezier(t, p0, p1, p2, p3):
    x = p0[0]*(1-t)**3 + 3*p1[0]*t*(1-t)**2 + 3*p2[0]*t**2*(1-t) + p3[0]*t**3
    y = p0[1]*(1-t)**3 + 3*p1[1]*t*(1-t)**2 + 3*p2[1]*t**2*(1-t) + p3[1]*t**3
    return x, y

def bezier_tangent(t, p0, p1, p2, p3):
    dx = -3*p0[0]*(1-t)**2 + 3*p1[0]*(1-t)**2 - 6*p1[0]*t*(1-t) + \
         6*p2[0]*t*(1-t) - 3*p2[0]*t**2 + 3*p3[0]*t**2
    dy = -3*p0[1]*(1-t)**2 + 3*p1[1]*(1-t)**2 - 6*p1[1]*t*(1-t) + \
         6*p2[1]*t*(1-t) - 3*p2[1]*t**2 + 3*p3[1]*t**2
    return dx, dy

def get_text_width(text, font, tracking=0):
    if not text: return 0, []
    try:
        char_widths = [font.getlength(ch) for ch in text]
        total_width = sum(char_widths) + tracking * (len(text) - 1)
        return total_width, char_widths
    except Exception:
        return 0, []

# ==========================================
# 坐标转换工具函数
# ==========================================
def rotate_coords_90deg(coords, img_width, img_height):
    """
    将坐标顺时针旋转90度
    假设输入格式为 [x1, y1, x2, y2] (左上角和右下角)
    """
    x1, y1, x2, y2 = coords
    # 顺时针旋转90度的变换: (x, y) -> (y, img_width - x)
    new_x1 = y1
    new_y1 = img_width - x2  # 注意这里使用x2来保持矩形方向
    new_x2 = y2
    new_y2 = img_width - x1  # 注意这里使用x1来保持矩形方向
    return [new_x1, new_y1, new_x2, new_y2]

def flip_coords_y_axis(coords, img_height):
    """
    沿y轴翻转坐标（如果坐标系原点在下方）
    假设输入格式为 [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = coords
    new_y1 = img_height - y2  # 注意这里使用y2
    new_y2 = img_height - y1  # 注意这里使用y1
    return [x1, new_y1, x2, new_y2]

def convert_coords(coords, img_width, img_height, rotation_type="rotate_90"):
    """
    根据指定的旋转类型转换坐标
    """
    if rotation_type == "rotate_90":
        return rotate_coords_90deg(coords, img_width, img_height)
    elif rotation_type == "flip_y":
        return flip_coords_y_axis(coords, img_height)
    else:
        return coords
    
# ==========================================
# 1. 配置类
# ==========================================
class Config:
    # 路径配置
    OUTPUT_DIR = "/app/cold1/code/texteditRoPE/data_construction_overall/dataset_output_v2"
    FONT_DIR = "/app/cold1/fontsttf"  # 请确保该目录下有 .ttf 字体文件
    
    # 源数据路径 (按你要求的路径)
    ORIGIN_IMG_DIR = "/app/cold1/datasets/Poster100K/poster100k/images"
    # JSON 路径推断：假设在 images 同级的 mask_regions 文件夹，或者就在 images 里
    # 根据你的描述 "每个图片一个json文件"，这里假设和图片在同一级或者有一个对应目录
    # 这里假设有一个 labels 目录，如果json和图片在一起，修改为和 IMG_DIR 一致
    ORIGIN_JSON_DIR = "/app/cold1/datasets/Poster100K/poster100k/mask_regions" 
    
    # 模型路径
    MODEL_QWEN_EDIT = "/app/cold1/Qwen-Image-Edit-2509" 
    MODEL_STYLE_GEN = "/app/cold1/Ovis-Image"
    
    # 固定分辨率
    TARGET_SIZE = (832, 1248)
    TARGET_WIDTH=1000
    TARGET_HEIGHT=1500
    MAX_ITEMS = 5000 
    
    # 生成参数
    PROB_CURVED = 0.5 
    
    # 风格提示词库
    STYLE_PROMPTS = [
        "Neon light style text, glowing, cyberpunk",
        "3D metallic gold text, shiny, luxury",
        "Chalkboard writing style, dusty, rough",
        "Wooden texture text, carved, natural",
        "Ice block text, frozen, blue, transparent",
        "Fire flame text, burning, hot",
        "Graffiti spray paint style, colorful, street art",
        "Cloudy smoke text, airy, soft white",
        "Liquid metal text, chrome, reflective",
        "Floral typography, flowers, nature style"
    ]
    
    # 词库
    WORD_LIST = [
        "SALENEW", "OPENOFF", "HOTBEST", "竹影扫阶尘不动", "Kernel panic", "螺蛳粉配冰啤酒", "Echo in the canyon",
        "FOOD", "COOL", "FASTSUPER", "MEGA", "BIG", "DEAL", "SHOPBUY"
    ]

# ==========================================
# 2. 核心模块：渲染器
# ==========================================
class SyntheticRenderer:
    def __init__(self, font_dir):
        self.fonts = glob.glob(os.path.join(font_dir, "*.ttf"))
        if not self.fonts:
            print(f"Warning: No fonts found in {font_dir}, using default.")
            self.fonts = []

    def get_random_font(self, size):
        if self.fonts:
            font_path = random.choice(self.fonts)
            try:
                return ImageFont.truetype(font_path, size)
            except:
                pass
        return ImageFont.load_default()

    # --- 碰撞与边界检测 ---
    def check_boundary(self, box, img_w, img_h):
        if not box: return False
        x1, y1, x2, y2 = box
        return x1 >= 0 and y1 >= 0 and x2 <= img_w and y2 <= img_h

    def check_overlap(self, box1, box_list):
        if not box1: return False
        for box2 in box_list:
            # (x1, y1, x2, y2)
            if not (box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3]):
                return True # Overlap detected
        return False

    # --- 绘制逻辑 ---
    def draw_text_on_layer(self, img_size, text, font, tracking, is_curved, occupied_boxes,text_color="black"):
        """
        在透明图层上绘制文字，返回 (Layer, Mask, BBox)
        """
        w, h = img_size
        temp_layer = Image.new("RGBA", img_size, (0,0,0,0))
        
        # 尝试多次寻找不碰撞的位置
        for attempt in range(20):
            # 清空
            temp_layer.paste((0,0,0,0), (0,0,w,h))
            
            # 随机参数
            if is_curved:
                curve_intensity = random.uniform(0.1, 0.6)
                bbox = self._draw_curved_core(temp_layer, text, font, tracking, curve_intensity)
            else:
                bbox = self._draw_normal_core(temp_layer, text, font, tracking,text_color)
            
            if bbox:
                # 检查边界
                if not self.check_boundary(bbox, w, h): continue
                # 检查碰撞
                if self.check_overlap(bbox, occupied_boxes): continue
                
                # 成功
                # 创建mask - 使用完整的bbox矩形区域
                mask = Image.new("L", img_size, 0)
                mask_draw = ImageDraw.Draw(mask)
                mask_draw.rectangle(bbox, fill=255)
                return temp_layer, mask, bbox
        
        return None, None, None

    def _draw_normal_core(self, layer, text, font, tracking, color="black"):
        total_w, _ = get_text_width(text, font, tracking)
        if total_w == 0: return None
        ascent, descent = font.getmetrics()
        text_h = ascent + descent
        
        temp_w = int(total_w * 1.2)
        temp_h = int(text_h * 1.5)
        txt_img = Image.new('RGBA', (temp_w, temp_h), (255, 255, 255, 0))
        d = ImageDraw.Draw(txt_img)
        # 这里使用了传入的 color
        d.text(((temp_w - total_w)//2, (temp_h - text_h)//2), text, font=font, fill=color)
        
        angle = random.uniform(-45, 45)
        rotated = txt_img.rotate(angle, resample=Image.BICUBIC, expand=True)
        
        img_w, img_h = layer.size
        w_final, h_final = rotated.size
        
        margin = 50
        max_x = max(margin, img_w - w_final - margin)
        max_y = max(margin, img_h - h_final - margin)
        
        paste_x = random.randint(margin, max_x)
        paste_y = random.randint(margin, max_y)
        
        layer.paste(rotated, (paste_x, paste_y), mask=rotated)
        return (paste_x, paste_y, paste_x + w_final, paste_y + h_final)

    def _draw_curved_core(self, layer, text, font, tracking, curve_intensity, color="black"):
        total_w, char_widths = get_text_width(text, font, tracking)
        if total_w == 0: return None
        
        img_w, img_h = layer.size
        margin = 50
        
        max_start_x = max(margin, img_w - int(total_w) - margin)
        start_x = random.randint(margin, max_start_x)
        start_y = random.randint(img_h // 4, img_h // 4 * 3)
        
        arch_h = total_w * curve_intensity * random.choice([1, -1])
        
        p0 = (start_x, start_y)
        p3 = (start_x + total_w, start_y)
        p1 = (start_x + total_w * random.uniform(0.2, 0.4), start_y - arch_h)
        p2 = (start_x + total_w * random.uniform(0.6, 0.8), start_y - arch_h)
        
        real_min_x, real_min_y = float('inf'), float('inf')
        real_max_x, real_max_y = float('-inf'), float('-inf')
        drawn = False
        current_dist = 0
        
        for i, ch in enumerate(text):
            w = char_widths[i]
            t = (current_dist + w/2) / max(1, total_w)
            t = max(0, min(1, t))
            
            pt = bezier(t, p0, p1, p2, p3)
            tg = bezier_tangent(t, p0, p1, p2, p3)
            angle = math.degrees(math.atan2(tg[1], tg[0]))
            
            char_size = int(font.size * 2.5)
            char_img = Image.new("RGBA", (char_size, char_size), (255,255,255,0))
            d = ImageDraw.Draw(char_img)
            bbox = font.getbbox(ch)
            if not bbox: continue
            text_h = bbox[3] - bbox[1]
            # 这里使用了传入的 color
            d.text((char_size//2 - w/2, char_size//2 - text_h//2 - bbox[1]), ch, font=font, fill=color)
            
            rotated = char_img.rotate(-angle, resample=Image.BICUBIC, expand=True)
            paste_x = int(pt[0] - rotated.width / 2)
            paste_y = int(pt[1] - rotated.height / 2)
            
            layer.paste(rotated, (paste_x, paste_y), mask=rotated)
            
            real_min_x = min(real_min_x, paste_x)
            real_min_y = min(real_min_y, paste_y)
            real_max_x = max(real_max_x, paste_x + rotated.width)
            real_max_y = max(real_max_y, paste_y + rotated.height)
            
            drawn = True
            current_dist += w + tracking
            
        if not drawn: return None
        return (int(real_min_x), int(real_min_y), int(real_max_x), int(real_max_y))

    # --- 生成 Content 图 (白底黑字) ---
    def render_content_image(self, img_size, text):
        """
        生成白底黑字的 Content 图，概率弯曲/直排
        """
        c_img = Image.new("RGB", img_size, (255, 255, 255))
        temp_layer = Image.new("RGBA", img_size, (0,0,0,0))
        
        font_size = random.randint(120, 160)
        font = self.get_random_font(font_size)
        tracking = random.randint(0, 8)
        is_curved = random.random() < Config.PROB_CURVED
        
        # 使用空列表作为 occupied_boxes，因为 content 图本身就是白的，随便放
        layer, _, _ = self.draw_text_on_layer(img_size, text, font, tracking, is_curved, [])
        
        if layer:
            c_img.paste(layer, (0,0), mask=layer)
            
        return c_img

# ==========================================
# 3. 主流水线
# ==========================================
class DataPipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.renderer = SyntheticRenderer(Config.FONT_DIR)
        
        # 输出目录结构
        self.sub_dirs = {
            "origin": os.path.join(Config.OUTPUT_DIR, "origins"),
            "mask": os.path.join(Config.OUTPUT_DIR, "masks"),
            "gt": os.path.join(Config.OUTPUT_DIR, "groundtruths"),
            "content": os.path.join(Config.OUTPUT_DIR, "contents"),
            "style": os.path.join(Config.OUTPUT_DIR, "styles"),
            "meta": os.path.join(Config.OUTPUT_DIR, "meta")
        }
        for d in self.sub_dirs.values(): os.makedirs(d, exist_ok=True)
            
        self.setup_models()
        
    def setup_models(self):
        print("Loading Models...")
        # Qwen-Edit
        self.pipe_edit = DiffusionPipeline.from_pretrained(
            Config.MODEL_QWEN_EDIT, torch_dtype=torch.bfloat16, device_map="balanced"
        )
        # Ovis-Gen (指定 cuda:3)
        self.pipe_style = OvisImagePipeline.from_pretrained(
            Config.MODEL_STYLE_GEN, torch_dtype=torch.bfloat16
        )
        self.pipe_style.to("cuda:1")

    def process_dataset(self):
        # 遍历图片
        img_pattern = os.path.join(Config.ORIGIN_IMG_DIR, "*.jpg")
        img_files = sorted(glob.glob(img_pattern))
        print(f"Found {len(img_files)} images in {Config.ORIGIN_IMG_DIR}")

        metadata_log = []
        blank_img = Image.new("RGB", Config.TARGET_SIZE, (255, 255, 255))
        
        for idx, img_path in enumerate(img_files):
            if idx >= Config.MAX_ITEMS: break
            
            filename = os.path.basename(img_path)
            item_id =filename.split("_")[-1].split(".")[0]
            print(f"Processing [{idx}] {item_id}...")

            # -----------------------------------------------------------------
            # 1. 读取 I0 和 M1 (JSON)
            # -----------------------------------------------------------------
            json_filename = f"poster_{item_id}_mask_regions.json"
            json_path = os.path.join(Config.ORIGIN_JSON_DIR, json_filename)
            
            if not os.path.exists(json_path): 
                # 尝试 fallback 目录结构
                json_path = os.path.join(Config.ORIGIN_IMG_DIR, json_filename)
                if not os.path.exists(json_path): continue

            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    mask_data = json.load(f)
                    raw_masks = json.loads(mask_data) if isinstance(mask_data, str) else mask_data
            except: continue

            # 读取原图并 Resize
            try:
                I0 = Image.open(img_path).convert("RGB").resize(Config.TARGET_SIZE)
            except: continue
            
            # 筛选
            if not raw_masks or len(raw_masks) > 4: continue

       
            
            # 统一 Resize
            I0 = I0.resize(Config.TARGET_SIZE, Image.BILINEAR)
            #M1 = M1_orig.resize(Config.TARGET_SIZE, Image.NEAREST)

            # -----------------------------------------------------------------
            # Group 1: 消除 (I0 -> I1)
            # -----------------------------------------------------------------
            gen = torch.Generator(self.device).manual_seed(42)
            try:
                I1 = self.pipe_edit(
                    prompt="去除图片中所有文字和字符。", 
                    image=I0, 
                    num_inference_steps=30, generator=gen
                ).images[0]
            except Exception as e:
                print(f"Error Gen I1: {e}")
                continue

            # 保存 Group 1: (I0, M1, C1=White, S1=White, I1)
            #self.save_group(item_id, "01", I0, M1, blank_img, blank_img, I1)

            # -----------------------------------------------------------------
            # Group 2: 中间态构造 (I1 -> I2 -> G2)
            # -----------------------------------------------------------------
            occupied_boxes_I2 = []
            
            # 2.1 在 I1 上渲染假字 -> I2, M2
            # "随机位置上用提供的字体随机选一个字体随机大小渲染，注意要检测碰撞"
            fake_text_2 = random.choice(Config.WORD_LIST)
            font_size = random.randint(120, 160)
            font = self.renderer.get_random_font(font_size)
            
            # 渲染 I2 的假字图层
            random_color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255), 255)
            fake_layer_2, fake_mask_2, bbox_2 = self.renderer.draw_text_on_layer(
                Config.TARGET_SIZE, fake_text_2, font, tracking=5, is_curved=False, 
                occupied_boxes=[], text_color=random_color # 传入随机颜色
            )
            
            if not fake_layer_2: continue # 失败则跳过
            occupied_boxes_I2.append(bbox_2)
            
            I2 = I1.copy()
            I2.paste(fake_layer_2, (0,0), mask=fake_layer_2)
            M2 = Image.new("L", Config.TARGET_SIZE, 0)
            M2.paste(fake_mask_2, (0,0), mask=fake_mask_2)

            # 2.2 构造 C2 (I2 的白底图，随机添加文本框)
            target_text_2 = random.choice(Config.WORD_LIST)
            C2 = self.renderer.render_content_image(Config.TARGET_SIZE, target_text_2)
            
            # 2.3 构造 S2 (风格图)
            style_prompt_2 = random.choice(Config.STYLE_PROMPTS)
            prompt_style=f"在白色背景上用{style_prompt_2}的风格绘制文字{random.choice(Config.WORD_LIST)}，让字体风格符合描述"
            S2 = self.pipe_style(prompt=prompt_style,num_inference_steps=30).images[0]
            
            # 2.4 生成 GT (G2) -> "给qwen传入I1、C2"
            prompt_edit_2 = f"在第一张图上把第二张图的文字用{style_prompt_2}风格的字体绘制，保持布局不变"
            try:
                G2 = self.pipe_edit(
                    prompt=prompt_edit_2,
                    image=[I1, C2], 
                    num_inference_steps=30
                ).images[0]
            except: continue
            
            self.save_group(item_id, "01", I2, M2, blank_img, blank_img, I1)

            # 保存 Group 2: (I2, M2, C2, S2, G2)
            self.save_group(item_id, "02", I2, M2, C2, S2, G2, 
                            remove=[fake_text_2], add=[target_text_2])

            # -----------------------------------------------------------------
            # Group 3: 中间态构造 Round 2 (I2 -> I3 -> G3)
            # -----------------------------------------------------------------
            # 3.1 在 I2 上渲染假字 -> I3, M3 (位置不能与 I2 上已有的碰撞)
            fake_text_3 = random.choice(Config.WORD_LIST)
            font_size_3 = random.randint(120, 160)
            font_3 = self.renderer.get_random_font(font_size_3)
            
            random_color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255), 255)
            fake_layer_3, fake_mask_3, bbox_3 = self.renderer.draw_text_on_layer(
                Config.TARGET_SIZE, fake_text_3, font_3, tracking=5, is_curved=False, occupied_boxes=occupied_boxes_I2,text_color=random_color 
            )
            
            if not fake_layer_3: continue 
            
            I3 = I2.copy()
            I3.paste(fake_layer_3, (0,0), mask=fake_layer_3)
            M3 = Image.new("L", Config.TARGET_SIZE, 0)
            M3.paste(fake_mask_3, (0,0), mask=fake_mask_3)

            # 3.2 构造 C3
            target_text_3 = random.choice(Config.WORD_LIST)
            C3 = self.renderer.render_content_image(Config.TARGET_SIZE, target_text_3)
            
            # 3.3 构造 S3 (其实是 S2 风格，但为了完整性生成新的S3或复用)
            # 描述说 "生成一张风格图S2"，这里可能是指复用S2或者生成新的，假设生成新的
            style_prompt_3 = random.choice(Config.STYLE_PROMPTS)
            S3 = self.pipe_style(prompt=style_prompt_3).images[0]
            
            # 3.4 生成 GT (G3) -> "给qwen传入I2、C3"
            prompt_edit_3 = f"在第一张图上用{style_prompt_3}的字体绘制图二中的文字，保持布局不变"
            try:
                G3 = self.pipe_edit(
                    prompt=prompt_edit_3,
                    image=[I2, C3], 
                    num_inference_steps=30
                ).images[0]
            except: continue

            # 保存 Group 3: (I3, M3, C3, S3, G3)
            self.save_group(item_id, "03", I3, M3, C3, S3, G3,
                            remove=[fake_text_3], add=[target_text_3])

            # -----------------------------------------------------------------
            # Group 4: 纯添加 (I1 -> G4)
            # -----------------------------------------------------------------
            # 4.1 Mask M4 (全黑)
            M4 = Image.new("L", Config.TARGET_SIZE, 0)
            
            # 4.2 C4
            target_text_4 = random.choice(Config.WORD_LIST)
            C4 = self.renderer.render_content_image(Config.TARGET_SIZE, target_text_4)
            
            # 4.3 S4
            style_prompt_4 = random.choice(Config.STYLE_PROMPTS)
            S4 = self.pipe_style(prompt=style_prompt_4).images[0]
            
            # 4.4 GT (G4) -> "给qwen传入I1、C4"
            prompt_edit_4 = f"在第一张图上用{style_prompt_4}的字体绘制图二中的文字，保持布局不变"
            try:
                G4 = self.pipe_edit(
                    prompt=prompt_edit_4,
                    image=[I1, C4], 
                    num_inference_steps=30
                ).images[0]
            except: continue

            # 保存 Group 4: (I1, M4, C4, S4, G4)
            self.save_group(item_id, "04", I1, M4, C4, S4, G4,
                            remove=[], add=[target_text_4])

            metadata_log.append({"id": item_id, "status": "success"})

        # End Loop
        with open(os.path.join(Config.OUTPUT_DIR, "dataset_meta.json"), "w") as f:
            json.dump(metadata_log, f, indent=4)
        print("Done.")

    def save_group(self, item_id, task_type, img, mask, content, style, gt, remove=[], add=[]):
        base_name = f"{item_id}{task_type}"
        files = {}
        
        # Save Images
        img.save(os.path.join(self.sub_dirs["origin"], f"{base_name}_origin.jpg"))
        mask.save(os.path.join(self.sub_dirs["mask"], f"{base_name}_mask.png"))
        content.save(os.path.join(self.sub_dirs["content"], f"{base_name}_content.jpg"))
        style.save(os.path.join(self.sub_dirs["style"], f"{base_name}_style.jpg"))
        gt.save(os.path.join(self.sub_dirs["gt"], f"{base_name}_groundtruth.jpg"))
        
        # Build Instruction
        parts = []
        if remove: parts.append(f"消除图中文字{'、'.join(remove)}")
        if add: parts.append(f"使用参考风格添加文字{'、'.join(add)}")
        # 特例：纯消除 Task 1
        if task_type == "01": parts = ["消除图中的文字"]
        
        instr = "，".join(parts)
        
        # Save Txt
        with open(os.path.join(self.sub_dirs["gt"], f"{base_name}_groundtruth.txt"), "w") as f:
            f.write(instr)
            
        # Save Meta
        meta = {
            "id": item_id,
            "task": task_type,
            "instruction": instr,
            "files": {
                "origin": f"{base_name}_origin.jpg",
                "mask": f"{base_name}_mask.png",
                "content": f"{base_name}_content.jpg",
                "style": f"{base_name}_style.jpg",
                "gt": f"{base_name}_groundtruth.jpg"
            }
        }
        with open(os.path.join(self.sub_dirs["meta"], f"{base_name}_meta.json"), "w") as f:
            json.dump(meta, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    DataPipeline().process_dataset()