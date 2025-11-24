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

content_image = Image.open("/app/code/texteditRoPE/train_data_dir/content_images/img1.jpg").convert("RGB")
content_image = Image.open("/app/code/texteditRoPE/train_data_dir/style_images/style1.jpg").convert("RGB")
prompt = "把文字\"knight\"的样式改成\"BBQ\"的样式，保持文字内容不变"
# 配置输入参数
inputs = {
    "image": [content_image,content_image],
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 50,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}

# 执行图像编辑
with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("/app/code/texteditRoPE/results/风格控制knight原版.png")
    #print(f"Edited image saved at: {output_path}")
    