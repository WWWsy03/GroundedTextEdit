# train_style_control.py
import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available
from PIL import Image # 导入 PIL 用于加载图像

from style_transfer_pipeline import QwenImageEditPlusPipelineWithStyleControl
if is_wandb_available():
    import wandb


logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Style Control on Qwen-Image-Edit")
    # --- 基础路径和模型配置 ---
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/app/cold1/Qwen-Image-Edit-2509",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files. One of [None, fp16, bf16, fp32].",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="qwenimage-style-control-output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    # --- 数据集配置 ---
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Expected structure:"
            " train_data_dir/"
            " ├── content_images/ (e.g., img1.jpg, img2.jpg, ...)"
            " ├── style_images/ (e.g., style1.jpg, style2.jpg, ...)"
            " └── prompts.txt (one prompt per line, order matching images)"
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    # --- 训练配置 ---
    parser.add_argument("--resolution", type=int, default=1024, help="Training resolution.")
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4, # 通常对新增层使用较小的学习率
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    # --- 验证和日志配置 ---
    parser.add_argument(
        "--validation_prompt", type=str, default=None, help="A prompt that is used during validation."
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run validation every X epochs. Validation consists of generating images with the validation prompt."
        ),
    )
    parser.add_argument(
        "--validation_content_image_path", # 新增
        type=str,
        default=None,
        help="Path to a content image for validation.",
    )
    parser.add_argument(
        "--validation_style_image_path", # 新增
        type=str,
        default=None,
        help="Path to a style image for validation.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args


# --- 自定义数据集类 ---
class StyleTransferDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, resolution=1024):
        super().__init__()
        self.root_dir = root_dir
        self.resolution = resolution

        content_img_dir = os.path.join(root_dir, "content_images")
        style_img_dir = os.path.join(root_dir, "style_images")
        prompt_file = os.path.join(root_dir, "prompts.txt")

        self.content_imgs = sorted([f for f in os.listdir(content_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.style_imgs = sorted([f for f in os.listdir(style_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        with open(prompt_file, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]

        assert len(self.content_imgs) == len(self.style_imgs) == len(self.prompts), \
               "Counts of content images, style images, and prompts must match."

        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) # 归一化到 [-1, 1]
        ])

    def __len__(self):
        return len(self.content_imgs)

    def __getitem__(self, index):
        content_img_path = os.path.join(self.root_dir, "content_images", self.content_imgs[index])
        style_img_path = os.path.join(self.root_dir, "style_images", self.style_imgs[index])
        prompt = self.prompts[index]

        content_img = Image.open(content_img_path).convert("RGB")
        style_img = Image.open(style_img_path).convert("RGB")

        content_img = self.transform(content_img)
        style_img = self.transform(style_img)

        return {"content_image": content_img, "style_image": style_img, "prompt": prompt}


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # --- 1. 加载 Pipeline 和 冻结模型 ---
    logger.info("Loading pipeline...")
    pipeline = QwenImageEditPlusPipelineWithStyleControl.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.bfloat16, # 或 torch.float16
        revision=args.revision,
        variant=args.variant,
    )

    # 冻结主要模型组件的参数
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.transformer.requires_grad_(False) # 整个 transformer 冻结
    # pipeline.scheduler 和 pipeline.processor 通常也不需要训练

    # --- 2. 确认 Processor 已设置 ---
    # 你的 pipeline.__init__ 应该已经将 transformer_blocks 的 processor 设置为 QwenDoubleStreamAttnProcessor2_0WithStyleControl
    # 确保这部分在 pipeline 加载后已经执行
    # 你可以通过检查 pipeline.transformer.transformer_blocks[0].attn.processor 的类型来验证
    # print(type(pipeline.transformer.transformer_blocks[0].attn.processor)) # Should be QwenDoubleStreamAttnProcessor2_0WithStyleControl

    # --- 3. 设置优化器 (只训练 style_k_proj 和 style_v_proj) ---
    # 收集所有 processor 中的 style_k_proj 和 style_v_proj 参数
    trainable_params = []
    for block in pipeline.transformer.transformer_blocks:
        processor = block.attn.processor
        if hasattr(processor, 'style_k_proj') and hasattr(processor, 'style_v_proj'):
            trainable_params.extend(list(processor.style_k_proj.parameters()))
            trainable_params.extend(list(processor.style_v_proj.parameters()))

    if not trainable_params:
         raise ValueError("No trainable parameters found in style control processors. Check pipeline initialization.")

    optimizer_cls = torch.optim.AdamW
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            raise ValueError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")

    optimizer = optimizer_cls(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # --- 4. 设置学习率调度器 ---
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # --- 5. 加载数据集 ---
    logger.info("Loading dataset...")
    train_dataset = StyleTransferDataset(root_dir=args.train_data_dir, resolution=args.resolution)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # --- 6. 准备 Accelerate ---
    pipeline, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        pipeline, optimizer, train_dataloader, lr_scheduler
    )

    # 如果使用梯度检查点，设置模型
    if args.gradient_checkpointing:
        pipeline.enable_gradient_checkpointing() # 这个方法需要在 prepare 之后调用，具体实现取决于 pipeline 结构

    # --- 7. 设置训练步数 ---
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # After len(train_dataloader) we need to do equivalent of ceil.
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    # 这里可以添加从检查点恢复训练的逻辑

    # Also move the alpha and gamma tensors to accelerator.device
    # (if using SNR gamma, etc.)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        pipeline.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(pipeline):
                # --- 数据准备 ---
                content_images = batch["content_image"].to(accelerator.device, dtype=pipeline.vae.dtype) # [B, C, H, W]
                style_images = batch["style_image"].to(accelerator.device, dtype=pipeline.vae.dtype)     # [B, C, H, W]
                prompts = batch["prompt"] # List of strings

                # 这里需要实现前向传播逻辑
                # 通常，风格控制训练会涉及到 DDPM/DDIM 噪声预测
                # 你需要一个 Scheduler (如 DDPMScheduler) 来添加噪声和计算 loss
                # 伪代码示意：
                # 1. Encode content_images using VAE to get content_latents
                # 2. Sample noise and timestep
                # 3. Add noise to content_latents
                # 4. Encode prompts using text_encoder
                # 5. Pack content_latents (and potentially style_latents if they were part of the input space)
                # 6. Call pipeline.transformer with noisy_latents, timesteps, prompt_embeds, attention_kwargs containing style_image_latents
                # 7. Get model_pred
                # 8. Calculate loss (e.g., MSE between model_pred and target noise/velocity)
                # 9. Backpropagate using accelerator.backward(loss)
                # 10. optimizer.step()
                # 11. lr_scheduler.step()
                # 12. optimizer.zero_grad(set_to_none=True)

                # 注意：pipeline 的 __call__ 方法是用于推理的，不适合直接用于训练循环
                # 你需要手动实现训练所需的 forward/backward 步骤，可能需要访问 pipeline 内部的组件（如 vae, text_encoder, transformer, scheduler）

                # 示例（不完整）：
                # scheduler = pipeline.scheduler # 或者重新加载一个 DDPMScheduler
                # vae = pipeline.vae
                # text_encoder = pipeline.text_encoder
                # processor = pipeline.transformer.transformer_blocks[...].attn.processor # Not directly usable in training loop

                # # 1. VAE Encoding
                # content_latents = vae.encode(content_images).latent_dist.sample() # Or mode()
                # content_latents = (content_latents - vae.config.latents_mean) / vae.config.latents_std
                # # Pack latents if necessary, similar to prepare_latents in pipeline
                # content_latents_packed = pipeline._pack_latents(...) # You'd need access to this or reimplement

                # # 2. Add Noise
                # noise = torch.randn_like(content_latents_packed)
                # timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (content_latents_packed.shape[0],), device=content_latents_packed.device)
                # timesteps = timesteps.long()
                # noisy_latents_packed = scheduler.add_noise(content_latents_packed, noise, timesteps)

                # # 3. Text Encoding (using pipeline's method or similar)
                # prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(prompts, ...) # Need to adapt for training batch

                # # 4. Style Image Processing (similar to pipeline's _encode_style_image or adapted)
                # style_image_latents = ... # Process style_images using VAE and pack, then project using ImageProjModel if needed
                # # For direct style latents input, you might pack the raw style_image pixel values after normalization
                # # This part depends heavily on how you want the style to interact internally
                # # The current processor takes style_image_latents (post-VAE) and injects into noise query
                # # You need to ensure the dimensions match expectation (L_style_patches, style_context_dim)
                # # and that noise_patches_length is passed correctly.

                # # 5. Call Transformer (manually, not pipeline.__call__)
                # model_pred = pipeline.transformer(
                #     hidden_states=noisy_latents_packed, # [B, L_noise, C_packed]
                #     timestep=timesteps / 1000, # Adjust based on scheduler
                #     encoder_hidden_states=prompt_embeds, # [B, L_txt, C_txt]
                #     encoder_hidden_states_mask=prompt_embeds_mask, # [B, L_txt]
                #     # ... other required args like img_shapes, txt_seq_lens
                #     attention_kwargs={
                #         "style_image_latents": style_image_latents, # [B, L_style, style_context_dim]
                #         "noise_patches_length": L_noise, # Scalar, int
                #         "style_scale": 1.0, # Scalar, float
                #         # ... other style related kwargs if needed by processor
                #     },
                #     return_dict=False,
                # )[0] # [B, L_noise, C_packed]

                # # 6. Calculate Loss
                # # Assuming predicting noise (eps)
                # loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                # # 7. Backpropagate
                # accelerator.backward(loss)
                # if accelerator.sync_gradients:
                #     params_to_clip = trainable_params # Only clip the params we're training
                #     accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                # optimizer.step()
                # lr_scheduler.step()
                # optimizer.zero_grad(set_to_none=True)

                # train_loss += loss.detach().item()

                # # Log step
                # if accelerator.sync_gradients:
                #     progress_bar.update(1)
                #     global_step += 1
                #     accelerator.log({"train_loss": train_loss / args.gradient_accumulation_steps, "step": global_step}, step=global_step)
                #     train_loss = 0.0

                # if global_step >= args.max_train_steps:
                #     break

                # Placeholder: Implement the actual training step logic here
                # This is complex because the pipeline is designed for inference, not training loops
                # You'll likely need to dissect the pipeline's forward pass components
                # and manually execute them within the training loop
                pass # Replace with actual training logic

        # --- 每个 epoch 后的验证 ---
        # if accelerator.is_main_process: # Only validate on main process
        #     if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
        #         logger.info(f"Running validation... \n Generating {args.num_validation_images} images with prompt: {args.validation_prompt}.")
        #         # Disable gradient computation for validation
        #         pipeline.eval()
        #         # ... (validation logic similar to inference, using fixed content/style images if needed)
        #         pipeline.train() # Switch back to train mode

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Save pipeline
        pipeline.save_pretrained(args.output_dir)

        # Save state dict of trainable parameters (style_k_proj, style_v_proj)
        # Collect state dict
        style_control_state_dict = {}
        unwrapped_pipeline = accelerator.unwrap_model(pipeline)
        for i, block in enumerate(unwrapped_pipeline.transformer.transformer_blocks):
            processor = block.attn.processor
            if hasattr(processor, 'style_k_proj') and hasattr(processor, 'style_v_proj'):
                for name, param in processor.style_k_proj.named_parameters():
                    style_control_state_dict[f"transformer_blocks.{i}.attn.processor.style_k_proj.{name}"] = param
                for name, param in processor.style_v_proj.named_parameters():
                    style_control_state_dict[f"transformer_blocks.{i}.attn.processor.style_v_proj.{name}"] = param

        # Save the specific state dict
        torch.save(style_control_state_dict, os.path.join(args.output_dir, "style_control_weights.safetensors")) # Or .bin

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()
