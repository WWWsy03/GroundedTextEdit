import torch
from PIL import Image
from grounded_pipeline import MyGroundedQwenPipeline # <-- 导入 v9 pipeline
import time

# --- 1. 定义您的实验 ---
experiments = [
    {
        "name": "condition用噪声区域编码",
        "switch_fraction": 0.1,
        "reverse_logic": False
    },
    # {
    #     "name": "layout_first5_20pct",
    #     "switch_fraction": 0.2,
    #     "reverse_logic": False
    # },
    # {
    #     "name": "layout_first5_30pct",
    #     "switch_fraction": 0.3,
    #     "reverse_logic": False
    # },
    # {
    #     "name": "layout_first5_40pct",
    #     "switch_fraction": 0.4,
    #     "reverse_logic": False
    # },
    # {
    #     "name": "layout_first5_50pct",
    #     "switch_fraction": 0.5,
    #     "reverse_logic": False
    # },
    # {
    #     "name": "layout_first5_60pct",
    #     "switch_fraction": 0.6,
    #     "reverse_logic": False
    # },
    # {
    #     "name": "layout_first5_70pct",
    #     "switch_fraction": 0.7,
    #     "reverse_logic": False
    # },
    # {
    #     "name": "layout_first5_80pct",
    #     "switch_fraction": 0.8,
    #     "reverse_logic": False
    # },
    # {
    #     "name": "layout_first5_90pct",
    #     "switch_fraction": 0.9,
    #     "reverse_logic": False
    # },
    # {
    #     "name": "layout_first5_100pct",
    #     "switch_fraction": 1,
    #     "reverse_logic": False
    # },
    # {
    #     "name": "style_first5_10pct_REVERSED",
    #     "switch_fraction": 0.1,
    #     "reverse_logic": True
    # },
    # {
    #     "name": "style_first5_20pct_REVERSED",
    #     "switch_fraction": 0.2,
    #     "reverse_logic": True
    # },
    # {
    #     "name": "style_first5_30pct_REVERSED",
    #     "switch_fraction": 0.3,
    #     "reverse_logic": True
    # },
    # {
    #     "name": "style_first5_40pct_REVERSED",
    #     "switch_fraction": 0.4,
    #     "reverse_logic": True
    # },
    # {
    #     "name": "style_first5_50pct_REVERSED",
    #     "switch_fraction": 0.5,
    #     "reverse_logic": True
    # },
    # {
    #     "name": "style_first5_60pct_REVERSED",
    #     "switch_fraction": 0.6,
    #     "reverse_logic": True
    # },
    # {
    #     "name": "style_first5_70pct_REVERSED",
    #     "switch_fraction": 0.7,
    #     "reverse_logic": True
    # },
    # {
    #     "name": "style_first5_80pct_REVERSED",
    #     "switch_fraction": 0.8,
    #     "reverse_logic": True
    # },
    # {
    #     "name": "style_first5_90pct_REVERSED",
    #     "switch_fraction": 0.9,
    #     "reverse_logic": True
    # },
    # {
    #     "name": "style_first5_100pct_REVERSED",
    #     "switch_fraction": 1,
    #     "reverse_logic": True
    # },
]

# --- 2. 加载模型 (一次即可) ---
MODEL_PATH = "/app/cold1/Qwen-Image-Edit-2509"
print(f"Loading model from {MODEL_PATH}...")

pipeline = MyGroundedQwenPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="balanced"  # 自动分配到多个GPU
)
pipeline.set_progress_bar_config(disable=None)
print("Model loaded.")

# --- 3. 加载图像 (一次即可) ---
print("Loading images...")
image_ori = Image.open("/app/code/texteditRoPE/samples/time.png").convert("RGB")
image_mask = Image.open("/app/code/texteditRoPE/samples/generated_mask.png").convert("L")
image_con = Image.open("/app/code/texteditRoPE/samples/time_con.png").convert("RGB").resize((900, 1350))

# --- 4. 循环执行所有实验 ---
for exp in experiments:
    print(f"\n--- Running Experiment: {exp['name']} ---")
    start_time = time.time()
    
    # 配置输入参数
    inputs = {
        "image": [image_ori, image_con],
        "erase_mask": image_mask,
        "prompt": "把图中文字\"The Wheel of Time\"改成\"时间之轮\"",
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 50, # 建议使用 50 步以获得更稳定的结果
        "guidance_scale": 1.0,
        "num_images_per_prompt": 1,
        
        # --- v9: 传入动态控制参数 ---
        #"control_switch_fraction": exp["switch_fraction"],
        #"control_reverse_logic": exp["reverse_logic"]
    }

    # 执行图像编辑
    with torch.inference_mode():
        output = pipeline(**inputs)
        output_image = output.images[0]
    
    end_time = time.time()
    
    # 保存结果
    output_filename = f"output_{exp['name']}.png"
    output_image.save(output_filename)
    print(f"✅ Experiment '{exp['name']}' complete in {end_time - start_time:.2f}s. Saved to {output_filename}")

print("\n--- All experiments complete. ---")