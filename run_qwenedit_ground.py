import torch
from PIL import Image
from grounded_pipeline import MyGroundedQwenPipeline
MODEL_PATH="/app/cold1/Qwen-Image-Edit-2509"
pipeline = MyGroundedQwenPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="balanced"  # 自动分配到多个GPU
)

pipeline.set_progress_bar_config(disable=None)


image_ori = Image.open("/app/code/texteditRoPE/samples/time.png").convert("RGB")
image_mask= Image.open("/app/code/texteditRoPE/samples/generated_mask.png").convert("L")
image_con=Image.open("/app/code/texteditRoPE/samples/time_con.png").convert("RGB").resize((900,1350))

# 配置输入参数
inputs = {
    "image": [image_ori,image_con],
    "erase_mask": image_mask,
    "prompt": "把图中文字\"The Wheel of Time\"改成\"时间之轮\"", # 注意：prompt 中的引号可能需要转义，例如 "改成\"西瓜火火茶\""
    # "generator": torch.manual_seed(0), # 如果需要固定随机种子
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 10,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1
}




# 执行图像编辑
with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("test5_50%_fix_inverse.png")
    #print(f"Edited image saved at: {output_path}")
    