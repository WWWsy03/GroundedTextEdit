from style_transfer_pipeline import QwenImageEditPlusPipelineWithStyleControl
#from style_transfer_processor import QwenDoubleStreamAttnProcessor2_0WithStyleControl
from PIL import Image
import torch

# 1. 加载 Qwen 模型基础组件
pipe = QwenImageEditPlusPipelineWithStyleControl.from_pretrained(
    "/app/cold1/Qwen-Image-Edit-2509", 
    torch_dtype=torch.bfloat16,
    device_map="balanced"
)

pipe.set_progress_bar_config(disable=None)
# 3. 创建风格投影模


# 加载图像
content_image = Image.open("/app/code/texteditRoPE/assets/example1.jpg").convert("RGB")
style_image = Image.open("/app/cold1/fonts/bench/ATR-bench/multi_letters/blue_lightning_knight.png").convert("RGB")
prompt = "把图中文字'一起瓜分夏天的快乐'改成'Lets share the joy of summer'"
#style_image = style_image.resize((content_image.width, content_image.height))
print(f"Style image size: {style_image.size}")
inputs = {
    "image": [content_image,style_image],
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "height": 1024,#暂时固定分辨率，不然噪声对于风格图的查询维度不匹配
    "width": 1024,
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 50,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
    "style_scale": 3.0,
    #"style_image": style_image, 
}

# 生成图像
image = pipe(**inputs).images[0]
image.save("output_with_style.png")