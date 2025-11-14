
from typing import Optional, List, Union, Dict, Any, Callable
from PIL import Image
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

# 导入你刚创建的 Processor
from style_transfer_processor import QwenDoubleStreamAttnProcessor2_0WithStyleControl

# Copyright 2025 Qwen-Image Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import QwenImageLoraLoaderMixin
from diffusers.models import AutoencoderKLQwenImage, QwenImageTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from PIL import Image
        >>> from diffusers import QwenImageEditPlusPipeline
        >>> from diffusers.utils import load_image

        >>> pipe = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png"
        ... ).convert("RGB")
        >>> prompt = (
        ...     "Make Pikachu hold a sign that says 'Qwen Edit is awesome', yarn art style, detailed, vibrant colors"
        ... )
        >>> # Depending on the variant being used, the pipeline call will slightly vary.
        >>> # Refer to the pipeline documentation for more details.
        >>> image = pipe(image, prompt, num_inference_steps=50).images[0]
        >>> image.save("qwenimage_edit_plus.png")
        ```
"""

CONDITION_IMAGE_SIZE = 384 * 384
VAE_IMAGE_SIZE = 1024 * 1024


class ImageProjModel(nn.Module):
    """图像投影模型，用于将 CLIP 特征投影到 Transformer 兼容的维度。"""
    def __init__(self, clip_embeddings_dim: int, cross_attention_dim: int, num_tokens: int = 4):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.proj = nn.Linear(clip_embeddings_dim, self.num_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds: torch.Tensor):
        # image_embeds: [B, clip_embeddings_dim]
        x = self.proj(image_embeds) # [B, num_tokens * cross_attention_dim]
        x = x.view(-1, self.num_tokens, self.cross_attention_dim) # [B, num_tokens, cross_attention_dim]
        x = self.norm(x) # [B, num_tokens, cross_attention_dim]
        return x
    
    
# Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    ):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
    ):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
    ):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return width, height

class QwenImageEditPlusPipelineWithStyleControl(DiffusionPipeline, QwenImageLoraLoaderMixin):
    r"""
    The Qwen-Image-Edit pipeline for image editing with style control.
    """
    model_cpu_offload_seq = "text_encoder->image_encoder->transformer->vae" # 更新 offload 序列
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLQwenImage,
        text_encoder: Qwen2_5_VLForConditionalGeneration,
        tokenizer: Qwen2Tokenizer,
        processor: Qwen2VLProcessor,
        transformer: QwenImageTransformer2DModel,
        # 新增 CLIP 相关组件
        image_encoder: CLIPVisionModelWithProjection = None,
        clip_image_processor: CLIPImageProcessor = None,
        # 新增风格投影模型
        style_proj_model: ImageProjModel = None,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            processor=processor,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder=image_encoder, # 注册 image_encoder
            style_proj_model=style_proj_model, # 注册 style_proj_model
        )
        # 注意：clip_image_processor 不是 nn.Module，所以不注册到 modules
        self.clip_image_processor = clip_image_processor # 作为属性保存

        self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.latent_channels = self.vae.config.z_dim if getattr(self, "vae", None) else 16
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.tokenizer_max_length = 1024
        self.prompt_template_encode = "..."
        self.prompt_template_encode_start_idx = 64
        self.default_sample_size = 128

        print("🔥开始热插拔 Attention Processors...")
        # 遍历 Transformer (DiT) 的所有 Block
        if hasattr(self, "transformer") and hasattr(self.transformer, "transformer_blocks"):
            total_blocks = len(self.transformer.transformer_blocks)
            print(f"✅找到了 {total_blocks} 个 blocks。")

            # --- 获取维度信息 ---
            # 1. 获取 style_hidden_dim (假设它与 transformer 的内部维度相关)
            # 检查 config 中是否有 hidden_size
            if hasattr(self.transformer.config, 'hidden_size'):
                 style_hidden_dim = self.transformer.config['hidden_size'] # 从 FrozenDict 获取
                 print(f"✅ 从 transformer.config['hidden_size'] 获取 style_hidden_dim: {style_hidden_dim}")
            elif hasattr(self.transformer.config, 'num_attention_heads') and hasattr(self.transformer.config, 'attention_head_dim'):
                 # 计算方式：num_heads * head_dim
                 style_hidden_dim = self.transformer.config['num_attention_heads'] * self.transformer.config['attention_head_dim']
                 print(f"✅ 从 transformer.config['num_attention_heads'] * config['attention_head_dim'] 计算 style_hidden_dim: {style_hidden_dim}")
            else:
                 raise ValueError("无法从 transformer.config 确定 style_hidden_dim。请检查模型配置。")

            # 2. 获取 style_context_dim (CLIP 编码器的输出维度)
            # 这需要在加载 image_encoder 后才能确定
            if self.image_encoder is not None:
                style_context_dim = self.image_encoder.config.projection_dim
                print(f"✅ 从 image_encoder.config.projection_dim 获取 style_context_dim: {style_context_dim}")
            else:
                 # 如果 image_encoder 尚未加载，需要在设置时传入或稍后确定
                 print("⚠️ image_encoder 未在初始化时提供，style_context_dim 需在设置处理器前确定。")
                 style_context_dim = None # 或者传入一个默认值，但这通常不安全

            for i, block in enumerate(self.transformer.transformer_blocks):
                # 为每个 block 创建并分配带风格控制的 processor
                # 注意：这里 processor 不再需要预先知道 style_context_dim 和 style_hidden_dim
                # 这些信息将通过 attention_kwargs 传递
                block.attn.processor = QwenDoubleStreamAttnProcessor2_0WithStyleControl(
                    style_context_dim=16,
                    style_hidden_dim=3072
                )

        else:
            print("⚠️ 警告: 找不到 transformer.transformer_blocks。热插拔失败。")

        print("✅ 热插拔完成！")
        
    def set_encoder(self):
        if hasattr(self, "transformer") and hasattr(self.transformer, "transformer_blocks"):
            total_blocks = len(self.transformer.transformer_blocks)
            print(f"✅找到了 {total_blocks} 个 blocks。")

            # --- 获取维度信息 ---
            # 1. 获取 style_hidden_dim (假设它与 transformer 的内部维度相关)
            # 检查 config 中是否有 hidden_size
            if hasattr(self.transformer.config, 'hidden_size'):
                 style_hidden_dim = self.transformer.config['hidden_size'] # 从 FrozenDict 获取
                 print(f"✅ 从 transformer.config['hidden_size'] 获取 style_hidden_dim: {style_hidden_dim}")
            elif hasattr(self.transformer.config, 'num_attention_heads') and hasattr(self.transformer.config, 'attention_head_dim'):
                 # 计算方式：num_heads * head_dim
                 style_hidden_dim = self.transformer.config['num_attention_heads'] * self.transformer.config['attention_head_dim']
                 print(f"✅ 从 transformer.config['num_attention_heads'] * config['attention_head_dim'] 计算 style_hidden_dim: {style_hidden_dim}")
            else:
                 raise ValueError("无法从 transformer.config 确定 style_hidden_dim。请检查模型配置。")

            # 2. 获取 style_context_dim (CLIP 编码器的输出维度)
            # 这需要在加载 image_encoder 后才能确定
            if self.image_encoder is not None:
                style_context_dim = self.image_encoder.config.projection_dim
                print(f"✅ 从 image_encoder.config.projection_dim 获取 style_context_dim: {style_context_dim}")
            else:
                 # 如果 image_encoder 尚未加载，需要在设置时传入或稍后确定
                 print("⚠️ image_encoder 未在初始化时提供，style_context_dim 需在设置处理器前确定。")
                 style_context_dim = None # 或者传入一个默认值，但这通常不安全
        
        
        
        

    # --- 以下方法与原 pipeline 基本相同 ---
    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result

    def _get_qwen_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        image: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt
        img_prompt_template = "Picture {}: <tool_call><tool_call><tool_call>"
        if isinstance(image, list):
            base_img_prompt = ""
            for i, img in enumerate(image):
                base_img_prompt += img_prompt_template.format(i + 1)
        elif image is not None:
            base_img_prompt = img_prompt_template.format(1)
        else:
            base_img_prompt = ""
        template = self.prompt_template_encode
        drop_idx = self.prompt_template_encode_start_idx
        txt = [template.format(base_img_prompt + e) for e in prompt]
        model_inputs = self.processor(
            text=txt,
            images=image,
            padding=True,
            return_tensors="pt",
        ).to(device)
        outputs = self.text_encoder(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pixel_values=model_inputs.pixel_values,
            image_grid_thw=model_inputs.image_grid_thw,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        return prompt_embeds, encoder_attention_mask

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        image: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 1024,
        ):
        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]
        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(prompt, image, device)
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)
        return prompt_embeds, prompt_embeds_mask

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds_mask=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
        ):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}. Dimensions will be resized accordingly"
            )
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        if prompt_embeds is not None and prompt_embeds_mask is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `prompt_embeds_mask` also have to be passed. Make sure to generate `prompt_embeds_mask` from the same text encoder that was used to generate `prompt_embeds`."
            )
        if negative_prompt_embeds is not None and negative_prompt_embeds_mask is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_prompt_embeds_mask` also have to be passed. Make sure to generate `negative_prompt_embeds_mask` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )
        if max_sequence_length is not None and max_sequence_length > 1024:
            raise ValueError(f"`max_sequence_length` cannot be greater than 1024 but is {max_sequence_length}")

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))
        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)
        return latents

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i], sample_mode="argmax")
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator, sample_mode="argmax")
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        image_latents = (image_latents - latents_mean) / latents_std
        return image_latents

    def prepare_latents(
        self,
        images,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        ):
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (batch_size, 1, num_channels_latents, height, width)
        image_latents = None
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            all_image_latents = []
            for image in images:
                image = image.to(device=device, dtype=dtype)
                if image.shape[1] != self.latent_channels:
                    image_latents = self._encode_vae_image(image=image, generator=generator)
                else:
                    image_latents = image
                if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
                    additional_image_per_prompt = batch_size // image_latents.shape[0]
                    image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
                elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
                    raise ValueError(
                        f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
                    )
                else:
                    image_latents = torch.cat([image_latents], dim=0)
                image_latent_height, image_latent_width = image_latents.shape[3:]
                image_latents = self._pack_latents(
                    image_latents, batch_size, num_channels_latents, image_latent_height, image_latent_width
                )
                all_image_latents.append(image_latents)
            image_latents = torch.cat(all_image_latents, dim=1)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        else:
            latents = latents.to(device=device, dtype=dtype)
        return latents, image_latents

    # --- 新增：处理风格图像的方法 ---
    def _encode_style_image(self, style_image: Image.Image, device: torch.device, dtype: torch.dtype):
        """
        将 PIL 图像通过 CLIP 编码，并使用 ImageProjModel 投影。
        """
        if self.image_encoder is None or self.clip_image_processor is None or self.style_proj_model is None:
            raise ValueError("Style control requires image_encoder, clip_image_processor, and style_proj_model. Please initialize the pipeline with them.")

        # 1. 预处理图像
        # CLIP 预处理器通常会将图像调整到固定大小（如 224x224）
        style_image_inputs = self.clip_image_processor(images=style_image, return_tensors="pt")
        style_image_pixel_values = style_image_inputs.pixel_values.to(device=device, dtype=dtype) # [B, C, H, W]

        # 2. 使用 CLIP 视觉编码器提取特征
        with torch.no_grad():
            image_embeds = self.image_encoder(style_image_pixel_values).image_embeds # [B, clip_embeddings_dim]

        # 3. 使用 ImageProjModel 投影到 Transformer 兼容的维度
        style_image_proj = self.style_proj_model(image_embeds) # [B, num_tokens, cross_attention_dim]

        return style_image_proj # [B, num_tokens, cross_attention_dim]

    # --- 主要修改：__call__ 方法 ---
    @torch.no_grad()
    def __call__(
        self,
        image: Optional[PipelineImageInput] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        style_image: Optional[PipelineImageInput] = None, # 新增参数
        true_cfg_scale: float = 4.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: Optional[float] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None, # 用于传递 style_image_proj
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        style_scale: float = 1.0, # 新增：风格强度缩放
    ):
        image_size = image[-1].size if isinstance(image, list) else image.size
        calculated_width, calculated_height = calculate_dimensions(1024 * 1024, image_size[0] / image_size[1])
        height = height or calculated_height
        width = width or calculated_width
        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs or {}
        self._current_timestep = None
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device

        # --- 处理风格图像 ---
        style_image_proj = None
        if style_image is not None:
            if isinstance(style_image, list):
                # 如果是列表，只处理第一个
                style_image = style_image[0]
            if isinstance(style_image, torch.Tensor):
                 # 如果已经是 tensor，假设它已经是预处理好的 CLIP 特征
                 # 你需要确保 tensor 的形状和 dtype 正确，例如 [B, num_tokens, cross_attention_dim]
                 style_image_proj = style_image
            else:
                # 如果是 PIL Image
                style_image_proj = self._encode_style_image(style_image, device, prompt_embeds.dtype if prompt_embeds is not None else self.transformer.dtype)
                # 重复以匹配 batch_size
                if batch_size > style_image_proj.shape[0] and batch_size % style_image_proj.shape[0] == 0:
                    additional_style_per_prompt = batch_size // style_image_proj.shape[0]
                    style_image_proj = torch.cat([style_image_proj] * additional_style_per_prompt, dim=0)
                elif batch_size > style_image_proj.shape[0] and batch_size % style_image_proj.shape[0] != 0:
                    raise ValueError(
                        f"Cannot duplicate `style_image` of batch size {style_image_proj.shape[0]} to {batch_size} prompts."
                    )
            # 将 style_image_proj 添加到 attention_kwargs
            self._attention_kwargs["style_image_proj"] = style_image_proj
            self._attention_kwargs["style_scale"] = style_scale # 传递 style_scale


        # 3. Preprocess image (content image)
        if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
            if not isinstance(image, list):
                image = [image]
            condition_image_sizes = []
            condition_images = []
            vae_image_sizes = []
            vae_images = []
            for img in image:
                image_width, image_height = img.size
                condition_width, condition_height = calculate_dimensions(
                    CONDITION_IMAGE_SIZE, image_width / image_height
                )
                vae_width, vae_height = calculate_dimensions(VAE_IMAGE_SIZE, image_width / image_height)
                condition_image_sizes.append((condition_width, condition_height))
                vae_image_sizes.append((vae_width, vae_height))
                condition_images.append(self.image_processor.resize(img, condition_height, condition_width))
                vae_images.append(self.image_processor.preprocess(img, vae_height, vae_width).unsqueeze(2))

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        if true_cfg_scale > 1 and not has_neg_prompt:
            logger.warning(
                f"true_cfg_scale is passed as {true_cfg_scale}, but classifier-free guidance is not enabled since no negative_prompt is provided."
            )
        elif true_cfg_scale <= 1 and has_neg_prompt:
            logger.warning(
                " negative_prompt is passed but classifier-free guidance is not enabled since true_cfg_scale <= 1"
            )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            image=condition_images,
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                image=condition_images,
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        num_channels_latents = self.transformer.config.in_channels // 4
        latents, image_latents = self.prepare_latents(
            vae_images,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        img_shapes = [
            [
                (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2),
                *[
                    (1, vae_height // self.vae_scale_factor // 2, vae_width // self.vae_scale_factor // 2)
                    for vae_width, vae_height in vae_image_sizes
                ],
            ]
        ] * batch_size

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        if self.transformer.config.guidance_embeds and guidance_scale is None:
            raise ValueError("guidance_scale is required for guidance-distilled model.")
        elif self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        elif not self.transformer.config.guidance_embeds and guidance_scale is not None:
            logger.warning(
                f"guidance_scale is passed as {guidance_scale}, but ignored since the model is not guidance-distilled."
            )
            guidance = None
        elif not self.transformer.config.guidance_embeds and guidance_scale is None:
            guidance = None

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
        )

        # --- Denoising loop ---
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                self._current_timestep = t
                latent_model_input = latents
                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1)

                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                with self.transformer.cache_context("cond"):
                    # 将 attention_kwargs 传递给 transformer，它会传递给 processor
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        encoder_hidden_states=prompt_embeds,
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                        attention_kwargs=self._attention_kwargs, # 关键：传递 attention_kwargs
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred[:, : latents.size(1)]

                if do_true_cfg:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=negative_prompt_embeds_mask,
                            encoder_hidden_states=negative_prompt_embeds,
                            img_shapes=img_shapes,
                            txt_seq_lens=negative_txt_seq_lens,
                            attention_kwargs=self.attention_kwargs, # 传递给 uncond 分支
                            return_dict=False,
                        )[0]
                    neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                    comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm)

                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return QwenImagePipelineOutput(images=image)