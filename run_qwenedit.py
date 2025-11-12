import os
import json
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import QwenImageEditPlusPipeline,DiffusionPipeline
import random
import textwrap  # 用于自动换行
from PIL import Image, ImageDraw, ImageFont
import textwrap
import random
import math
from typing import List, Union

"""
运行 Qwen-Image-Edit-2509 原生 进行图像编辑"""
MODEL_PATH="/app/cold1/Qwen-Image-Edit-2509"
pipeline = QwenImageEditPlusPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="balanced"
)
# pipeline = QwenImageEditPlusPipeline.from_pretrained(
#     MODEL_PATH,
#     torch_dtype=torch.bfloat16,
#     #device_map="balanced"  # 自动分配到多个GPU
# )
# 如果有LoRA，可以加载LoRA
#lora_path = "/mnt/workspace/wsy/flymyai-lora-trainer/test_lora_saves_edit_512/checkpoint-28800"
#pipeline.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors")

pipeline.set_progress_bar_config(disable=None)

image_ori = Image.open("/app/code/texteditRoPE/samples/time.png").convert("RGB")
image_mask = Image.open("/app/code/texteditRoPE/samples/generated_mask.png").convert("L")
image_con = Image.open("/app/code/texteditRoPE/samples/time_con.png").convert("RGB").resize((900, 1350))
     
# 配置输入参数
inputs = {
    "image": [image_ori,image_con],
    "prompt": "把图1中文字'The WHELL OF TIME'改成'时间之轮'，文字布局与第二张图相同，文字样式与第一张图原本文字样式相同",
    #"generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 10,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}

# 执行图像编辑
with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("/app/code/texteditRoPE/results/输入两张图qwen原生.png")
    #print(f"Edited image saved at: {output_path}")
    