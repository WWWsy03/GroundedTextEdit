import argparse
import copy
from copy import deepcopy
import logging
import os
import shutil
import gc
import math
import torch
import numpy as np
from tqdm.auto import tqdm
from PIL import Image

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
import datasets
import diffusers
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers import (
    AutoencoderKLQwenImage,
    QwenImageTransformer2DModel,
    QwenImagePipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.loaders import AttnProcsLayers
from omegaconf import OmegaConf
import transformers
from optimum.quanto import quantize, qfloat8, freeze
import bitsandbytes as bnb

# PEFT for LoRA
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

# Custom Modules
# 假设你的环境里这两个模块的命名如下，根据提供的第二份代码保持一致
from style_transfer_pipeline_doublestyle import QwenImageEditPlusPipelineWithStyleControl
from style_transfer_processor_doubelstyle import QwenDoubleStreamAttnProcessor2_0WithStyleControl
from image_datasets.control_dataset import loader, image_resize

logger = get_logger(__name__, log_level="INFO")

TARGET_IMAGE_SIZE = 1024
STYLE_SCALE = 10000.0



def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height, None


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for LoRA + Style KV.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config",
    )
    args = parser.parse_args()
    return args.config

# Helper function to find LoRA processors
def lora_processors(model):
    processors = {}
    def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors):
        if 'lora' in name:
            processors[name] = module
        for sub_name, child in module.named_children():
            fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

    for name, module in model.named_children():
        fn_recursive_add_processors(name, module, processors)
    return processors

def get_sigmas(timesteps, noise_scheduler, accelerator, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
    timesteps = timesteps.to(accelerator.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

def main():
    args = OmegaConf.load(parse_args())
    #args.save_cache_on_disk = True
    args.precompute_text_embeddings = True
    args.precompute_image_embeddings = True

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # 1. Load VAE and Text Pipeline for Caching/Preprocessing
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
    )
    vae.to(accelerator.device, dtype=weight_dtype)
    
    text_encoding_pipeline = QwenImageEditPlusPipelineWithStyleControl.from_pretrained(
        args.pretrained_model_name_or_path, transformer=None, vae=vae, torch_dtype=weight_dtype
    )
    text_encoding_pipeline.to(accelerator.device)

    cached_text_embeddings = {}
    cache_dir = os.path.join(args.output_dir, "cache")
    txt_cache_dir = os.path.join(cache_dir, "text_embs")
    os.makedirs(txt_cache_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    img_cache_dir = os.path.join(cache_dir, "img_embs")
    os.makedirs(img_cache_dir, exist_ok=True)
    # if args.precompute_text_embeddings or args.precompute_image_embeddings:
    #     if accelerator.is_main_process:
    #         cache_dir = os.path.join(args.output_dir, "cache")
    #         txt_cache_dir = os.path.join(cache_dir, "text_embs")
    #         os.makedirs(txt_cache_dir, exist_ok=True)
    #         os.makedirs(cache_dir, exist_ok=True)
    #         img_cache_dir = os.path.join(cache_dir, "img_embs")
    #         os.makedirs(img_cache_dir, exist_ok=True)
    # if args.precompute_text_embeddings:
    #     with torch.no_grad():
    #         if args.save_cache_on_disk:
    #             print('Saving text embeddings to disk cache at ', txt_cache_dir)
    #             # txt_cache_dir = os.path.join(cache_dir, "text_embs")
    #             # os.makedirs(txt_cache_dir, exist_ok=True)
    #         else:
    #             cached_text_embeddings = {}
    #         for img_name in tqdm([i for i in os.listdir(args.data_config.control_dir) if ".png" in i or '.jpg' in i]):
    #             id=img_name.split('.')[0]
    #             print(f"id: {id}")
    #             img_path = os.path.join(args.data_config.control_dir, img_name)
    #             style_name = img_name.replace('img', 'style')
    #             style_img_path = os.path.join(args.data_config.style_dir, style_name)
    #             txt_path = os.path.join(args.data_config.img_dir, img_name.split('.')[0] + '.txt')

    #             img = Image.open(img_path).convert('RGB')
    #             style_img= Image.open(style_img_path).convert('RGB')
    #             calculated_width, calculated_height, _ = calculate_dimensions(1024 * 1024, img.size[0] / img.size[1])
    #             prompt_image = text_encoding_pipeline.image_processor.resize(img, calculated_height, calculated_width)
    #             style_prompt_image = text_encoding_pipeline.image_processor.resize(style_img, calculated_height, calculated_width)
                
    #             prompt = open(txt_path, encoding='utf-8').read()
    #             prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(
    #                 image=[prompt_image,style_prompt_image],
    #                 prompt=[prompt],
    #                 device=text_encoding_pipeline.device,
    #                 num_images_per_prompt=1,
    #                 max_sequence_length=1024,
    #             )
    #             if args.save_cache_on_disk:
                    
    #                 torch.save({'prompt_embeds': prompt_embeds[0].to('cpu'), 'prompt_embeds_mask': prompt_embeds_mask[0].to('cpu')}, os.path.join(txt_cache_dir, str(id)+ '.pt'))
    #             else:
    #                 cached_text_embeddings[img_name.split('.')[0] + '.txt'] = {'prompt_embeds': prompt_embeds[0].to('cpu'), 'prompt_embeds_mask': prompt_embeds_mask[0].to('cpu')}
    #         # compute empty embedding
    #             prompt_embeds_empty, prompt_embeds_mask_empty = text_encoding_pipeline.encode_prompt(
    #                 image=[prompt_image,style_prompt_image],
    #                 prompt=[' '],
    #                 device=text_encoding_pipeline.device,
    #                 num_images_per_prompt=1,
    #                 max_sequence_length=1024,
    #             )
    #             cached_text_embeddings[str(img_name.split('.')[0]) + '.txt' + 'empty_embedding'] = {'prompt_embeds': prompt_embeds_empty[0].to('cpu'), 'prompt_embeds_mask': prompt_embeds_mask_empty[0].to('cpu')}
                    


    

    cached_image_embeddings = {}
    img_cache_dir = os.path.join(cache_dir, "img_embs")
    os.makedirs(img_cache_dir, exist_ok=True)
    control_cache_dir= os.path.join(cache_dir, "img_embs_control")
    os.makedirs(control_cache_dir, exist_ok=True)
    cached_image_embeddings_control = {}
    # if args.precompute_image_embeddings:
    #     if args.save_cache_on_disk:
    #         print('Saving image embeddings to disk cache at ', img_cache_dir)
    #     else:
    #         cached_image_embeddings = {}
    #     with torch.no_grad():
    #         for img_name in tqdm([i for i in os.listdir(args.data_config.img_dir) if ".png" in i or ".jpg" in i]):
    #             img = Image.open(os.path.join(args.data_config.img_dir, img_name)).convert('RGB')
    #             id=img_name.split('.')[0]
    #             calculated_width, calculated_height, _ = calculate_dimensions(1024 * 1024, img.size[0] / img.size[1])
    #             img = text_encoding_pipeline.image_processor.resize(img, calculated_height, calculated_width)

    #             img = torch.from_numpy((np.array(img) / 127.5) - 1)
    #             img = img.permute(2, 0, 1).unsqueeze(0)
    #             pixel_values = img.unsqueeze(2)
    #             pixel_values = pixel_values.to(dtype=weight_dtype).to(accelerator.device)
        
    #             pixel_latents = vae.encode(pixel_values).latent_dist.sample().to('cpu')[0]
    #             if args.save_cache_on_disk:
    #                 torch.save(pixel_latents, os.path.join(img_cache_dir, str(id) + '.pt'))
    #                 del pixel_latents
    #             else:
    #                 cached_image_embeddings[img_name] = pixel_latents
    #     if args.save_cache_on_disk:
    #         # img_cache_dir = os.path.join(cache_dir, "img_embs_control")
    #         # os.makedirs(img_cache_dir, exist_ok=True)
    #         pass
    #     else:
    #         cached_image_embeddings_control = {}
    #     with torch.no_grad():
    #         for img_name in tqdm([i for i in os.listdir(args.data_config.control_dir) if ".png" in i or ".jpg" in i]):
    #             img = Image.open(os.path.join(args.data_config.control_dir, img_name)).convert('RGB')
    #             id=img_name.split('.')[0]
    #             calculated_width, calculated_height, _ = calculate_dimensions(1024 * 1024, img.size[0] / img.size[1])
    #             img = text_encoding_pipeline.image_processor.preprocess(img, calculated_width, calculated_width).unsqueeze(2)
    #             style_img_name = img_name.replace('img', 'style')
    #             style_img = Image.open(os.path.join(args.data_config.style_dir, style_img_name)).convert('RGB') 
    #             style_img = text_encoding_pipeline.image_processor.preprocess(style_img, calculated_height, calculated_width).unsqueeze(2)
        
    #             vae_images = [img, style_img]
                
    #             # pixel_values = img.unsqueeze(2)
    #             # pixel_values = pixel_values.to(dtype=weight_dtype).to(accelerator.device)
        
    #             # pixel_latents = vae.encode(pixel_values).latent_dist.sample().to('cpu')[0]
    #             #num_channels_latents = text_encoding_pipeline.transformer.config.in_channels // 4   是16 但是这个时候transformer还没挂上
    #             num_channels_latents=16
    #             latents, image_latents, L_noise, style_image_latents, style_start_idx, style_end_idx = text_encoding_pipeline.prepare_latents(
    #                 vae_images, # 传入包含 content 和 style 的图像列表
    #                 1,
    #                 num_channels_latents,
    #                 TARGET_IMAGE_SIZE,
    #                 TARGET_IMAGE_SIZE,
    #                 weight_dtype,
    #                 accelerator.device,
    #                 generator=None,
    #             )
    #             # 7. 保存所有数据到字典
    #             save_data = {
    #                 "image_latents": image_latents.cpu(),            # [1, Seq, C] (Content + Style 拼接好的)
    #                 "style_image_latents": style_image_latents.cpu(),# [1, Seq, C] (单独 Style 用于 KV)
    #                 "L_noise": L_noise,
    #                 "style_start_idx": style_start_idx,
    #                 "style_end_idx": style_end_idx,
    #             }
    #             if args.save_cache_on_disk:
    #                 torch.save(save_data, os.path.join(control_cache_dir, str(id) + '.pt'))
    #                 print( os.path.join(control_cache_dir, str(id) + '.pt'))
    #                 del image_latents
    #                 del latents 
    #             else:
    #                 print('caching control image embedding for ', img_name)
    #                 cached_image_embeddings_control[img_name] = save_data
    # Clean up memory after preprocessing preparation
    vae_config=vae.config
    vae_temperal_downsample=vae.temperal_downsample
    vae.to('cpu')
    text_encoding_pipeline.to("cpu")
    torch.cuda.empty_cache()
    del vae
    del text_encoding_pipeline
    gc.collect()
    print('Finished setting up embeddings cache paths.')

    # 2. Load Transformer
    flux_transformer = QwenImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
    )

    # Quantization logic
    if args.quantize:
        torch_dtype = weight_dtype
        device = accelerator.device
        all_blocks = list(flux_transformer.transformer_blocks)
        for block in tqdm(all_blocks, desc="Quantizing"):
            block.to(device, dtype=torch_dtype)
            quantize(block, weights=qfloat8)
            freeze(block)
            block.to('cpu')
        flux_transformer.to(device, dtype=torch_dtype)
        quantize(flux_transformer, weights=qfloat8)
        freeze(flux_transformer)

    # 3. Add LoRA Adapter
    # LoRA Target Modules from Script 2
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    # Adding adapter usually enables gradients for LoRA layers
    flux_transformer.add_adapter(lora_config)

    # 4. Replace Attention Processors with Style Control Processors
    # Logic from Script 1 to ensure dimensions are set correctly for the new KV layers
    style_context_dim = 64  # Based on Script 1 comments
    attn_processors = {}
    
    for name in flux_transformer.attn_processors.keys():
        style_hidden_dim = flux_transformer.config['num_attention_heads'] * flux_transformer.config['attention_head_dim']
        
        # Instantiate with dimensions so weights are created
        processor = QwenDoubleStreamAttnProcessor2_0WithStyleControl(
            style_context_dim=style_context_dim,
            style_hidden_dim=style_hidden_dim
        )
        attn_processors[name] = processor

    # Set the processors
    flux_transformer.set_attn_processor(attn_processors)

    # 5. Set Gradients (Hybrid Approach)
    flux_transformer.train()
    flux_transformer.enable_gradient_checkpointing()
    
    # Start by freezing everything
    flux_transformer.requires_grad_(False)

    # A. Unfreeze LoRA Layers
    for n, param in flux_transformer.named_parameters():
        if 'lora' in n:
            param.requires_grad = True

    # B. Unfreeze Style KV Layers (from the new Processors)
    for module in flux_transformer.modules():
        if isinstance(module, QwenDoubleStreamAttnProcessor2_0WithStyleControl):
            if hasattr(module, 'style_k_proj'):
                module.style_k_proj.requires_grad_(True)
            if hasattr(module, 'style_v_proj'):
                module.style_v_proj.requires_grad_(True)
            if hasattr(module, 'style_scale'):
                module.style_scale.requires_grad_(True)
                module.style_scale.data = module.style_scale.data.to(torch.float32)

    # 6. Calculate and Print Parameters
    trainable_params_list = list(filter(lambda p: p.requires_grad, flux_transformer.parameters()))
    total_trainable_params = sum(p.numel() for p in trainable_params_list)
    print(f"\n{'='*40}")
    print(f"Total Trainable Parameters (LoRA + Style KV): {total_trainable_params / 1e6:.4f} M")
    print(f"{'='*40}\n")

    # 7. Optimizer Setup
    optimizer_cls = torch.optim.AdamW
    if args.adam8bit:
        optimizer = bnb.optim.Adam8bit(
            trainable_params_list,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
        )
    else:
        optimizer = optimizer_cls(
            trainable_params_list,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    # Scheduler and Data
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    train_dataloader = loader(
        cached_text_embeddings=None, 
        cached_image_embeddings=None, 
        cached_image_embeddings_control=None,
        txt_cache_dir=txt_cache_dir,
        img_cache_dir=img_cache_dir,
        control_cache_dir=control_cache_dir,
        **args.data_config
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare with Accelerator
    flux_transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        flux_transformer, optimizer, train_dataloader, lr_scheduler
    )

    # Trackers
    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, {"test": None})

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info(f"Total train batch size = {total_batch_size}")

    global_step = 0
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    vae_scale_factor = 2 ** len(vae_temperal_downsample) # config check needed if this fails

    # Training Loop
    for epoch in range(100000):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(flux_transformer):
                # Data Unpacking (Using Script 1 logic to get style control inputs)
                if args.precompute_text_embeddings:
                    img, prompt_embeds, prompt_embeds_mask, control_img = batch
                    prompt_embeds = prompt_embeds.to(dtype=weight_dtype).to(accelerator.device)
                    prompt_embeds_mask = prompt_embeds_mask.to(dtype=torch.int32).to(accelerator.device)
                    
                    # Prepare attention_kwargs for the Custom Processor
                    attention_kwargs = {}
                    attention_kwargs["style_image_latents"] = control_img['style_image_latents'].to(dtype=weight_dtype).to(accelerator.device)
                    attention_kwargs["style_start_idx"] = control_img['L_noise'] + control_img['style_start_idx']
                    attention_kwargs["style_end_idx"] = control_img['L_noise'] + control_img['style_end_idx']
                    attention_kwargs["noise_patches_length"] = control_img['L_noise'] 
                    attention_kwargs["style_scale"] = STYLE_SCALE
                    
                    # The main content image latents
                    image_latents = control_img['image_latents'].to(dtype=weight_dtype).to(accelerator.device)
                else:
                    # Fallback logic not implemented fully in provided code
                    raise NotImplementedError("Requires precomputed embeddings logic.")

                # Noise generation
                pixel_latents = img.to(dtype=weight_dtype).to(accelerator.device)
                pixel_latents = pixel_latents.permute(0, 2, 1, 3, 4)
                
                # Use image_latents from control_img as the conditioning part for concatenation
                control_latents_input = image_latents 

                # Normalize latents
                latents_mean = torch.tensor(vae_config.latents_mean).view(1, 1, vae_config.z_dim, 1, 1).to(pixel_latents.device, pixel_latents.dtype)
                latents_std = 1.0 / torch.tensor(vae_config.latents_std).view(1, 1, vae_config.z_dim, 1, 1).to(pixel_latents.device, pixel_latents.dtype)
                pixel_latents = (pixel_latents - latents_mean) * latents_std
                
                bsz = pixel_latents.shape[0]
                noise = torch.randn_like(pixel_latents, device=accelerator.device, dtype=weight_dtype)
                
                # Timestep sampling
                u = compute_density_for_timestep_sampling(
                    weighting_scheme="none",
                    batch_size=bsz,
                    logit_mean=0.0,
                    logit_std=1.0,
                    mode_scale=1.29,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=pixel_latents.device)
                
                sigmas = get_sigmas(timesteps, noise_scheduler_copy, accelerator, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
                noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise
                
                # Pack Latents
                packed_noisy_model_input = QwenImageEditPlusPipelineWithStyleControl._pack_latents(
                    noisy_model_input,
                    bsz, 
                    noisy_model_input.shape[2],
                    noisy_model_input.shape[3],
                    noisy_model_input.shape[4],
                )
                
                # Concatenate with control latents (Logic from Script 1)
                packed_noisy_model_input_concated = torch.cat([packed_noisy_model_input, control_latents_input], dim=1)
                packed_noisy_model_input_concated.requires_grad_(True)
                
                # Img Shapes for RoPE
                img_shapes = [[(1, noisy_model_input.shape[3] // 2, noisy_model_input.shape[4] // 2),
                               (1, noisy_model_input.shape[3] // 2, noisy_model_input.shape[4] // 2),
                               (1, noisy_model_input.shape[3] // 2, noisy_model_input.shape[4] // 2)]] * bsz
                
                txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
                print(f"txt_seq{txt_seq_lens}")

                # Forward Pass
                model_pred = flux_transformer(
                    hidden_states=packed_noisy_model_input_concated,
                    timestep=timesteps / 1000,
                    guidance=None,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    attention_kwargs=attention_kwargs, # Pass style control args here
                    return_dict=False,
                )[0]
                
                # Slice output
                model_pred = model_pred[:, : packed_noisy_model_input.size(1)]
                
                # Unpack
                model_pred = QwenImageEditPlusPipelineWithStyleControl._unpack_latents(
                    model_pred,
                    height=noisy_model_input.shape[3] * vae_scale_factor,
                    width=noisy_model_input.shape[4] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )
                
                # Loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
                target = noise - pixel_latents
                target = target.permute(0, 2, 1, 3, 4)
                
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()
                
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(flux_transformer.parameters(), args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    # 1. 首先确保所有进程都运行到了这一步
                    accelerator.wait_for_everyone()
                    
                    # 2. 这里的路径生成虽然只需要主进程用，但在外面定义也没关系
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")

                    # 3. [关键修复] get_state_dict 必须在所有进程中调用！
                    # 只有这样，DeepSpeed 才能从各个 GPU 把碎片拼凑成完整的权重
                    # 注意：这会把完整模型加载到 CPU 内存，请确保内存足够
                    unwrapped_model = unwrap_model(flux_transformer)
                    full_state_dict = accelerator.get_state_dict(flux_transformer)

                    # 4. 只有主进程负责保存文件到磁盘
                    if accelerator.is_main_process:
                        os.makedirs(save_path, exist_ok=True)
                        
                        # --- Hybrid Saving Logic ---
                        
                        # A. 保存 LoRA 权重
                        # 使用 peft 的工具从 full_state_dict 中提取 LoRA 权重
                        flux_transformer_lora_state_dict = convert_state_dict_to_diffusers(
                                get_peft_model_state_dict(unwrapped_model, state_dict=full_state_dict)
                        )
                        QwenImagePipeline.save_lora_weights(
                            save_path,
                            flux_transformer_lora_state_dict,
                            safe_serialization=True,
                        )
                        
                        # B. 保存 Custom Style KV 权重
                        style_state_dict = {}
                        for k, v in full_state_dict.items():
                            # 筛选条件：包含 style 相关关键字，且不包含 lora
                            # (虽然 lora 已经被单独提取了，但在 raw state dict 里可能还存在，为了纯净建议过滤)
                            if ("style_k_proj" in k or "style_v_proj" in k or "style_scale" in k) and "lora" not in k:
                                style_state_dict[k] = v.cpu()
                        
                        from safetensors.torch import save_file
                        save_file(style_state_dict, os.path.join(save_path, "style_control_layers.safetensors"))
                        
                        logger.info(f"Saved LoRA and Style KV weights to {save_path}")
                    
                    # 5. [关键] 所有进程都要释放内存，因为 full_state_dict 在所有进程都存在
                    del full_state_dict
                    torch.cuda.empty_cache()
                    
                    # 再次同步，确保主进程保存完之前，其他进程不会开始下一轮训练（可选，但推荐）
                    accelerator.wait_for_everyone()

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
        
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == "__main__":
    main()