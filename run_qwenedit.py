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

origin_image = Image.open("/app/cold1/code/texteditRoPE/textEditing-test-case/2_origin.png").convert("RGB").resize((1024,1024))
content_image = Image.open("/app/cold1/code/texteditRoPE/textEditing-test-case/2_content.png").convert("RGB").resize((1024,1024))
style_image = Image.open("/app/cold1/code/texteditRoPE/textEditing-test-case/2_style.png").convert("RGB").resize((1024,1024))
mask_image = Image.open("/app/cold1/code/texteditRoPE/textEditing-test-case/2_mask.png").convert("RGB").resize((1024,1024))
prompt = "修改图一中的文字，将图一中mask区域所对应的文字去掉，并按照图三文字布局参考图四文字风格在图一上绘制文字Picnic Day"
# 配置输入参数
inputs = {
    "image": [origin_image,mask_image,content_image,style_image],
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
    output_image.save("/app/cold1/code/texteditRoPE/textEditing-test-case/test_results/test2_prompt2.png")
    #print(f"Edited image saved at: {output_path}")
    