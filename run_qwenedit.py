import torch
from PIL import Image
from grounded_pipeline_rope import MyGroundedQwenPipeline # <-- 导入 v12 pipeline
import time
import os

# --- 1. v12: 实验被简化为单次运行 ---
# (移除了循环)

# --- 2. 加载模型 (一次即可) ---
MODEL_PATH = "/app/cold1/Qwen-Image-Edit-2509"
print(f"Loading model from {MODEL_PATH}...")

# 确保导入的是您修改后的 Pipeline (v12)
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

output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
print(f"Results will be saved to {output_dir}")


exp_name = "v14_Pure_rope_only_ctrl_bg_offset"
print(f"\n--- Running Experiment: {exp_name} ---")
start_time = time.time()

# 配置输入参数
inputs = {
    "image": [image_ori, image_con],
    "erase_mask": image_mask, # v12: 移除
    "prompt": "把图中文字\"The Wheel of Time\"改成\"时间之轮\"",
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 10, 
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
    "generator": torch.Generator(device=pipeline.device).manual_seed(42) # 固定种子
    
    # --- v12: 移除所有控制参数 ---
    # "control_switch_fraction": ...,
    # "control_reverse_logic": ...
}

# 执行图像编辑
with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]

end_time = time.time()

# 保存结果
output_filename = os.path.join(output_dir, f"output_{exp_name}.png")
output_image.save(output_filename)
print(f"✅ Experiment '{exp_name}' complete in {end_time - start_time:.2f}s. Saved to {output_filename}")

print("\n--- v12 test complete. ---")