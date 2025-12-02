from style_transfer_pipeline_doublestyle import QwenImageEditPlusPipelineWithStyleControl
from style_transfer_processor_doubelstyle import QwenDoubleStreamAttnProcessor2_0WithStyleControl
from style_transformer_qwenimage_doublestyle import QwenImageTransformer2DModel
from PIL import Image
import torch
from safetensors.torch import load_file
from safetensors.torch import load_file

# custom_transformer = QwenImageTransformer2DModel.from_pretrained(
#     "/app/cold1/Qwen-Image-Edit-2509",  # 或者你的本地路径
#     subfolder="transformer",
#     torch_dtype=torch.bfloat16
# )

# 1. 加载 Qwen 模型基础组件
pipe = QwenImageEditPlusPipelineWithStyleControl.from_pretrained(
    "/app/cold1/Qwen-Image-Edit-2509", 
    #transformer=custom_transformer,
    torch_dtype=torch.bfloat16,
    device_map="balanced"
)

pipe.set_progress_bar_config(disable=None)
# 3. 创建风格投影模

# lora_path = "/app/cold1/code/texteditRoPE/qwenimage-style-rope-lora-large-scale/checkpoint-2500"
# pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors")


# checkpoint_path = "/app/cold1/code/texteditRoPE/qwenimage-style-rope-lora-large-scale/checkpoint-2500/style_control_layers.safetensors"

# print(f"正在加载 Style 权重: {checkpoint_path}")

# # 读取权重字典
# style_state_dict = load_file(checkpoint_path)

# # 【关键步骤】加载权重到 Transformer
# # strict=False 是必须的，因为 style_state_dict 只包含部分参数（style_k/v），
# # 而 transformer 包含所有参数。strict=False 允许只加载匹配的键。
# missing_keys, unexpected_keys = pipe.transformer.load_state_dict(style_state_dict, strict=False)

# 验证加载是否成功
# 我们期望 style_state_dict 里的所有键都被加载了，所以 unexpected_keys 应该为空（相对于 state_dict 而言）
# 但在这里 unexpected_keys 指的是 transformer 里有但 state_dict 里没有的键（这会很多，不用管）
# 我们主要关心的是：我们提供的权重是否都找到了对应的层。
# loaded_keys = style_state_dict.keys()
# print(f"成功加载了 {len(loaded_keys)} 个 Style 参数张量。")

# # 简单的验证打印
# if len(missing_keys) > 0:
#     # 只要 missing_keys 里不包含 'style_k_proj' 或 'style_v_proj' 就没事
#     style_missing = [k for k in missing_keys if "style_" in k]
#     if len(style_missing) > 0:
#         print(f"⚠️ 警告: 以下 Style 参数未能加载 (可能键名不匹配): {style_missing}")
#     else:
#         print("✅ 所有 Style Control 参数已成功注入模型！")
        
        
        
        
# 加载图像
origin_image = Image.open("/app/cold1/code/texteditRoPE/textEditing-test-case/2_origin.png").convert("RGB").resize((1024,1024))
content_image = Image.open("/app/cold1/code/texteditRoPE/textEditing-test-case/2_content.png").convert("RGB").resize((1024,1024))
style_image = Image.open("/app/cold1/code/texteditRoPE/textEditing-test-case/2_style.png").convert("RGB").resize((1024,1024))
mask_image = Image.open("/app/cold1/code/texteditRoPE/textEditing-test-case/2_mask.png").convert("RGB").resize((1024,1024))
prompt = "修改图一中的文字，将图一中mask区域所对应的文字去掉，并按照图三文字布局参考图四文字风格在图一上绘制文字Picnic Day"
#style_image = style_image.resize((content_image.width, content_image.height))
print(f"Style image size: {style_image.size}")
inputs = {
    "image": [origin_image,mask_image,content_image,style_image],
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "height": 1024,#暂时固定分辨率，不然噪声对于风格图的查询维度不匹配
    "width": 1024,
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 50,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
    "style_scale": 0.0,
    #"style_image": style_image, 
}

# 生成图像
image = pipe(**inputs).images[0]
image.save("/app/cold1/code/texteditRoPE/train_lora_kv_rope/overall-pipeline/test_results/test2_planB_scale0.png")