import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

# 复用 Pipeline 中的静态方法，放在外部方便调用
def pack_latents(latents, batch_size, num_channels_latents, height, width, vae_scale_factor=8):
    # Qwen-Image specific packing logic
    height_packed = 2 * (int(height) // (vae_scale_factor * 2))
    width_packed = 2 * (int(width) // (vae_scale_factor * 2))
    
    latents = latents.view(batch_size, num_channels_latents, height_packed // 2, 2, width_packed // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height_packed // 2) * (width_packed // 2), num_channels_latents * 4)
    return latents

class QwenEditDataset(Dataset):
    def __init__(self, data_root, tokenizer, size=512):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.size = size
        
        self.content_dir = os.path.join(data_root, "content_images")
        self.style_dir = os.path.join(data_root, "style_images")
        self.gt_dir = os.path.join(data_root, "ground_truth_images")
        self.prompt_path = os.path.join(data_root, "prompts.txt")
        
        # 假设文件名是对应的 (img1.jpg, style1.jpg, gt1.jpg)
        # 并且 prompts.txt 的每一行对应排序后的文件
        self.content_files = sorted(os.listdir(self.content_dir))
        self.style_files = sorted(os.listdir(self.style_dir))
        self.gt_files = sorted(os.listdir(self.gt_dir))
        
        with open(self.prompt_path, 'r', encoding='utf-8') as f:
            self.prompts = [line.strip() for line in f.readlines()]
            
        assert len(self.content_files) == len(self.style_files) == len(self.gt_files) == len(self.prompts), \
            "Error: Image directories and prompts.txt line count do not match."

        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), # Map to [-1, 1]
        ])

    def __len__(self):
        return len(self.content_files)

    def __getitem__(self, idx):
        content_img = Image.open(os.path.join(self.content_dir, self.content_files[idx])).convert("RGB")
        style_img = Image.open(os.path.join(self.style_dir, self.style_files[idx])).convert("RGB")
        gt_img = Image.open(os.path.join(self.gt_dir, self.gt_files[idx])).convert("RGB")
        prompt = self.prompts[idx]

        return {
            "content_pixel_values": self.transform(content_img),
            "style_pixel_values": self.transform(style_img),
            "gt_pixel_values": self.transform(gt_img),
            "prompt": prompt
        }

def collate_fn(batch):
    content_pixel_values = torch.stack([item["content_pixel_values"] for item in batch])
    style_pixel_values = torch.stack([item["style_pixel_values"] for item in batch])
    gt_pixel_values = torch.stack([item["gt_pixel_values"] for item in batch])
    prompts = [item["prompt"] for item in batch]
    return {
        "content_pixel_values": content_pixel_values,
        "style_pixel_values": style_pixel_values,
        "gt_pixel_values": gt_pixel_values,
        "prompts": prompts
    }