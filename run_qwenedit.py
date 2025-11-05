import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline,DiffusionPipeline
MODEL_PATH="/app/cold1/Qwen-Image-Edit-2509"
pipeline = QwenImageEditPlusPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="balanced"  # 自动分配到多个GPU
)
# 如果有LoRA，可以加载LoRA
#lora_path = "/mnt/workspace/wsy/flymyai-lora-trainer/test_lora_saves_edit_mini/checkpoint-4200"
#pipeline.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors")


pipeline.set_progress_bar_config(disable=None)

def callback_on_step_end(pipeline, step: int, timestep: int, callback_kwargs: dict):
    """
    修正后的回调函数
    Args:
        pipeline: Pipeline 实例 (self)
        step (int): 当前步骤索引 (从0开始)
        timestep (int): 当前时间步
        callback_kwargs (dict): 包含指定张量的字典
    """
    # 从 callback_kwargs 中获取张量
    latents = callback_kwargs.get("latents")
    # prompt_embeds = callback_kwargs.get("prompt_embeds") # 如果需要也可以获取

    print(f"Step {step}, Timestep: {timestep}")
    print(f"  Latents shape: {latents.shape}") # 打印形状
    print(f"  Latents dtype: {latents.dtype}") # 打印数据类型
    print(f"  Latents min/max: {latents.min().item():.4f} / {latents.max().item():.4f}") # 打印值的范围

    # --- 可选：可视化 latents ---
    # 注意：latents 是打包后的格式 (batch_size, num_patches, num_channels_latents*4)
    # 需要先 _unpack_latents，然后 _decode，最后 _postprocess
    # 这里为了简化，仅打印信息。如果要可视化，需要调用 pipeline 内部的相关函数。
    # 你可以将 latents 保存下来，稍后处理，或者直接在回调函数内部调用 pipeline 的方法。
    # 例如，如果你想在这里可视化，你需要：
    # 1. 从 pipeline 获取必要的参数 (height, width, vae_scale_factor 等)
    # 2. 调用 pipeline._unpack_latents, pipeline.vae.decode, pipeline.image_processor.postprocess
    # 这比较复杂，通常建议将 latents 保存到列表中，在 pipeline 调用结束后再处理。

    # 示例：保存当前 latents (不推荐在回调中大量保存，可能占用内存)
    # if step % 10 == 0: # 每10步保存一次
    #     torch.save(latents.cpu(), f"latents_step_{step}_timestep_{timestep}.pt")

    # 返回修改后的 callback_kwargs (如果需要修改 latents 或 prompt_embeds，可以在这里改)
    # 例如： callback_kwargs["latents"] = modified_latents
    return callback_kwargs



image = Image.open("/app/code/TextEditor-v2/images/example1.jpg").convert("RGB")
image_ori = Image.open("/app/code/texteditRoPE/samples/time.png").convert("RGB")
image_mask= Image.open("/app/code/texteditRoPE/samples/generated_mask.png").convert("L")
image_con=Image.open("/app/code/texteditRoPE/samples/time_con.png").convert("RGB").resize((900,1350))
       
# 配置输入参数
inputs = {
    "image": [image_ori,image_con],
    "prompt": "把图中文字\"西瓜冰冰茶\"改成\"西瓜火火茶\"", # 注意：prompt 中的引号可能需要转义，例如 "改成\"西瓜火火茶\""
    # "generator": torch.manual_seed(0), # 如果需要固定随机种子
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 5,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
    "callback_on_step_end": callback_on_step_end, # 添加回调函数
    "callback_on_step_end_tensor_inputs": ["latents", "prompt_embeds"], # 指定要传递给回调函数的张量
}


# 执行图像编辑
with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("/app/code/PosterTranslator_old/test2fly4200.png")
    #print(f"Edited image saved at: {output_path}")
    