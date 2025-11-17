from style_transfer_pipeline import QwenImageEditPlusPipelineWithStyleControl, ImageProjModel
#from style_transfer_processor import QwenDoubleStreamAttnProcessor2_0WithStyleControl
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from PIL import Image
import torch

# 1. 加载 Qwen 模型基础组件
pipe = QwenImageEditPlusPipelineWithStyleControl.from_pretrained(
    "/app/cold1/Qwen-Image-Edit-2509", 
    torch_dtype=torch.bfloat16,
    device="balanced"
)

# 2. 加载 CLIP 组件 (用于风格图像编码)
# 注意：选择合适的 CLIP 模型 ID，例如 "openai/clip-vit-large-patch14"
image_encoder_id = "/app/cold1/clip-vit-large-patch14"
image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_id, torch_dtype=torch.bfloat16)
clip_image_processor = CLIPImageProcessor.from_pretrained(image_encoder_id)
#pipe.to("cuda")
pipe.set_progress_bar_config(disable=None)
# 3. 创建风格投影模型
# clip_embeddings_dim: CLIP 模型的投影层输出维度 (e.g., 768 for vit-large-patch14)
# cross_attention_dim: Qwen Transformer 的内部维度 (e.g., pipe.transformer.config.hidden_size)
# num_tokens: 你想让风格图像编码成多少个 token (e.g., 4, as in IP-Adapter)
clip_embeddings_dim = image_encoder.config.projection_dim
cross_attention_dim = 768 # ?????这个应该是多少有待确认
num_tokens = 4 # 例如，使用 4 个 token

print("cross_attention_dim:", cross_attention_dim)
style_proj_model = ImageProjModel(
    clip_embeddings_dim=clip_embeddings_dim,
    cross_attention_dim=cross_attention_dim,
    num_tokens=num_tokens
)

# 4. 将 CLIP 和投影模型设置到 pipeline
pipe.image_encoder = image_encoder.to(pipe.device, dtype=pipe.transformer.dtype)
pipe.clip_image_processor = clip_image_processor
pipe.style_proj_model = style_proj_model.to(pipe.device, dtype=pipe.transformer.dtype)
pipe.set_encoder()


# 加载图像
content_image = Image.open("/app/code/GroundedTextEdit/assets/example1.jpg").convert("RGB")
style_image = Image.open("/app/cold1/fonts/bench/ATR-bench/multi_letters/blue_lightning_knight.png").convert("RGB")
prompt = "把图中文字'一起瓜分夏天的快乐'改成'Lets share the joy of summer'"

inputs = {
    "image": content_image,
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 50,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
    "style_image": style_image, 
}

# 生成图像
image = pipe(**inputs).images[0]
image.save("output_with_style.png")