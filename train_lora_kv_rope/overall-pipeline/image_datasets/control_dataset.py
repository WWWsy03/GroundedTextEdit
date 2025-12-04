import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random

def throw_one(probability: float) -> int:
    return 1 if random.random() < probability else 0


def image_resize(img, max_size=512):
    w, h = img.size
    if w >= h:
        new_w = max_size
        new_h = int((max_size / w) * h)
    else:
        new_h = max_size
        new_w = int((max_size / h) * w)
    return img.resize((new_w, new_h))

def c_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))

def crop_to_aspect_ratio(image, ratio="16:9"):
    width, height = image.size
    ratio_map = {
        "16:9": (16, 9),
        "4:3": (4, 3),
        "1:1": (1, 1)
    }
    target_w, target_h = ratio_map[ratio]
    target_ratio_value = target_w / target_h

    current_ratio = width / height

    if current_ratio > target_ratio_value:
        new_width = int(height * target_ratio_value)
        offset = (width - new_width) // 2
        crop_box = (offset, 0, offset + new_width, height)
    else:
        new_height = int(width / target_ratio_value)
        offset = (height - new_height) // 2
        crop_box = (0, offset, width, offset + new_height)

    cropped_img = image.crop(crop_box)
    return cropped_img


import os
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# 辅助函数：随机丢弃 (用于 Classifier-Free Guidance 训练)
def throw_one(rate):
    return random.random() < rate

# 辅助函数：图片缩放 (保持原有逻辑)
def image_resize(img, size):
    return img.resize((size, size), Image.BICUBIC)

# 辅助函数：裁剪 (保持原有逻辑)
def crop_to_aspect_ratio(image, ratio):
    # 这里是一个简化的占位，如果你有具体的裁剪逻辑请保留你原来的
    # 假设 ratio 是 "1:1", "16:9" 等
    w, h = image.size
    if ratio == "1:1":
        target_size = min(w, h)
        left = (w - target_size) / 2
        top = (h - target_size) / 2
        return image.crop((left, top, left + target_size, top + target_size))
    return image

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, img_size=512, caption_type='txt',
                 random_ratio=False, caption_dropout_rate=0.1, 
                 cached_text_embeddings=None, cached_image_embeddings=None, 
                 control_dir=None, cached_image_embeddings_control=None, style_dir=None,origin_dir=None,mask_dir=None,
                 # 新增：接收缓存目录路径
                 txt_cache_dir="/app/code/texteditRoPE/qwenimage-style-control-output/cache/text_embs", img_cache_dir="/app/code/texteditRoPE/qwenimage-style-control-output/cache/img_embs", control_cache_dir="/app/code/texteditRoPE/qwenimage-style-control-output/cache/img_embs_control"):
        
        self.images = [os.path.join(img_dir, i) for i in os.listdir(img_dir) if '.jpg' in i or '.png' in i]
        self.images.sort()
        self.img_size = img_size
        self.caption_type = caption_type
        self.random_ratio = random_ratio
        self.caption_dropout_rate = caption_dropout_rate
        self.control_dir = control_dir
        # self.origin_dir=origin_dir
        # self.mask_dir=mask_dir
        
        # 内存缓存字典
        self.cached_text_embeddings = cached_text_embeddings
        self.cached_image_embeddings = cached_image_embeddings
        self.cached_control_image_embeddings = cached_image_embeddings_control
        
        # 磁盘缓存目录
        self.txt_cache_dir = txt_cache_dir
        self.img_cache_dir = img_cache_dir
        self.control_cache_dir = control_cache_dir
        print(f"cache_dirs: txt: {self.txt_cache_dir}, img: {self.img_cache_dir}, control: {self.control_cache_dir}")
        
        # 简单的 Debug 打印
        # print(f"Dataset initialized. Txt Cache Dir: {self.txt_cache_dir}")

    def __len__(self):
        # 建议返回真实长度，如果你希望无限循环可以在 DataLoader 外层控制，
        # 或者保留你原来的 999999
        return len(self.images) 

    def __getitem__(self, idx):
        try:
            # 防止索引越界 (如果 len 返回的是 999999)
            if idx >= len(self.images):
                idx = random.randint(0, len(self.images) - 1)
                
            img_path = self.images[idx]
            # 获取纯文件名，例如 "img0.jpg"
            img_filename = img_path.split('/')[-1] 
            # 获取 ID，例如 "img0" (用于文本缓存文件名匹配)
            img_id = img_filename.split('.')[0]
            img_id=img_id.split('_')[0]

            # =======================================================
            # 1. 处理 GT Image (目标图)
            # =======================================================
            # 你的保存逻辑是: os.path.join(img_cache_dir, str(img_name) + '.pt') -> "img0.jpg.pt"
            gt_pt_filename = img_id + '.pt'
            img = None

            # A. 尝试从磁盘缓存读取
            if self.img_cache_dir and os.path.exists(os.path.join(self.img_cache_dir, gt_pt_filename)):
                img = torch.load(os.path.join(self.img_cache_dir, gt_pt_filename))
                # 去掉 Batch 维度 [1, C, H, W] -> [C, H, W]
                if isinstance(img, torch.Tensor) and img.dim() == 4 and img.shape[0] == 1:
                    img = img.squeeze(0)
                
            
            # B. 尝试从内存缓存读取
            elif self.cached_image_embeddings and img_filename in self.cached_image_embeddings:
                img = self.cached_image_embeddings[img_filename]
                if isinstance(img, torch.Tensor) and img.dim() == 4 and img.shape[0] == 1:
                    img = img.squeeze(0)
            
            # C. 实时处理 (回退方案)
            else:
                img_pil = Image.open(img_path).convert('RGB')
                if self.random_ratio:
                    ratio = random.choice(["16:9", "default", "1:1", "4:3"])
                    if ratio != "default":
                        img_pil = crop_to_aspect_ratio(img_pil, ratio)
                img_pil = image_resize(img_pil, self.img_size)
                w, h = img_pil.size
                new_w = (w // 32) * 32
                new_h = (h // 32) * 32
                img_pil = img_pil.resize((new_w, new_h))
                img = torch.from_numpy((np.array(img_pil) / 127.5) - 1)
                img = img.permute(2, 0, 1) # [C, H, W]

            # =======================================================
            # 2. 处理 Control Image (字典数据)
            # =======================================================
            # 你的保存逻辑同上: "img0.jpg.pt"
            control_img = None
            
            # A. 尝试从磁盘缓存读取
            if self.control_cache_dir and os.path.exists(os.path.join(self.control_cache_dir, gt_pt_filename)):
                data_dict = torch.load(os.path.join(self.control_cache_dir, gt_pt_filename))
                control_img = {}
                for k, v in data_dict.items():
                    # 遍历字典，如果是 Tensor 且第一维是 1，则 squeeze
                    if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == 1:
                        control_img[k] = v.squeeze(0)
                    else:
                        control_img[k] = v
                        
                #print(f"control_img keys from disk: {list(control_img.keys())}")

            # B. 尝试从内存缓存读取
            elif self.cached_control_image_embeddings and img_filename in self.cached_control_image_embeddings:
                data_dict = self.cached_control_image_embeddings[img_filename]
                control_img = {}
                for k, v in data_dict.items():
                    if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == 1:
                        control_img[k] = v.squeeze(0)
                    else:
                        control_img[k] = v

            # C. 实时处理 (Control 部分比较复杂，如果没缓存通常需要报错或写完整处理逻辑)
            else:
                # 这里简单回退到读取原图，但注意这不符合你 Pipeline 的 prepare_latents 需求
                # 如果必须实时训练，需在这里补充完整的 prepare_latents 逻辑
                # 为防止报错，这里返回一个简单的 Tensor，但建议确保缓存存在
                if self.control_dir:
                    ctrl_path = os.path.join(self.control_dir, img_filename)
                    if os.path.exists(ctrl_path):
                        c_img = Image.open(ctrl_path).convert('RGB')
                        c_img = image_resize(c_img, self.img_size)
                        # ... 省略 Resize ...
                        control_img = torch.from_numpy((np.array(c_img) / 127.5) - 1).permute(2, 0, 1)
                    else:
                        control_img = torch.zeros_like(img) # Dummy fallback
                else:
                    control_img = torch.zeros_like(img)

            # =======================================================
            # 3. 处理 Text Embeddings
            # =======================================================
            # 你的保存逻辑是: os.path.join(txt_cache_dir, str(id) + '.pt') -> "img0.pt"
            txt_pt_filename = img_id + '.pt'
            
            prompt_ret = None
            mask_ret = None
            return_str = False

            # A. 尝试从磁盘缓存读取
            if self.txt_cache_dir and os.path.exists(os.path.join(self.txt_cache_dir, txt_pt_filename)):
                
                if throw_one(self.caption_dropout_rate):
                    # 理想情况：加载 empty.pt
                    # 现实情况：你没存 empty.pt
                    # 补救：返回一个全零的 embedding 作为 Uncond (如果模型支持) 或者返回空字符串
                    # 这里演示读取正常 Embedding
                    txt_data = torch.load(os.path.join(self.txt_cache_dir, txt_pt_filename))
                    prompt_ret = txt_data['prompt_embeds']
                    mask_ret = txt_data['prompt_embeds_mask']
                    
                    # 如果一定要实现 dropout，可以将 prompt_ret 设为全零
                    # prompt_ret = torch.zeros_like(prompt_ret) 
                else:
                    txt_data = torch.load(os.path.join(self.txt_cache_dir, txt_pt_filename))
                    prompt_ret = txt_data['prompt_embeds']
                    mask_ret = txt_data['prompt_embeds_mask']

                # Squeeze Batch Dim
                if prompt_ret is not None and prompt_ret.shape[0] == 1:
                    prompt_ret = prompt_ret.squeeze(0)
                if mask_ret is not None and mask_ret.shape[0] == 1:
                    mask_ret = mask_ret.squeeze(0)

            # B. 尝试从内存读取
            elif self.cached_text_embeddings:
                txt_key = img_filename.split('.')[0] # img0
                # 这里你原代码有逻辑处理 empty_embedding
                if throw_one(self.caption_dropout_rate):
                     key = txt_key + '.txt' + 'empty_embedding'
                else:
                     key = txt_key + '.txt'
                
                if key in self.cached_text_embeddings:
                    prompt_ret = self.cached_text_embeddings[key]['prompt_embeds']
                    mask_ret = self.cached_text_embeddings[key]['prompt_embeds_mask']
                    if prompt_ret.shape[0] == 1: prompt_ret = prompt_ret.squeeze(0)
                    if mask_ret.shape[0] == 1: mask_ret = mask_ret.squeeze(0)
            
            # C. 实时读取文本文件
            else:
                txt_path = img_path.split('.')[0] + '.' + self.caption_type
                if os.path.exists(txt_path):
                    prompt_text = open(txt_path, encoding='utf-8').read()
                else:
                    prompt_text = ""
                    
                if throw_one(self.caption_dropout_rate):
                    prompt_ret = " " # 返回字符串
                    return_str = True
                else:
                    prompt_ret = prompt_text
                    return_str = True

            # =======================================================
            # 4. 返回结果
            # =======================================================
            # 如果是字符串 (没有 Embedding)，mask 返回 None 或 Dummy
            # 如果是 Embedding (Tensor)，正常返回
            
            if return_str:
                # 返回: img, prompt_string, control_img
                # 注意：你的 collate_fn 可能需要适配
                return img, prompt_ret, control_img
            else:
                # 返回: img, prompt_embeds, prompt_mask, control_img
                # 这里的 control_img 是一个字典 {'image_latents': ..., ...}
                #print(f"img{img}")
                return img, prompt_ret, mask_ret, control_img

        except Exception as e:
            #print(f"Error loading index {idx}: {e}")
            # 递归重试，但防止无限递归建议加个计数器，或者简单随机重试
            return self.__getitem__(random.randint(0, len(self.images) - 1))     

def collate_fn(batch):
    # batch 是一个 list，每个元素是 __getitem__ 返回的 tuple:
    # (img, prompt_embeds, prompt_mask, control_img_dict)
    
    imgs = []
    prompt_embeds_list = []
    prompt_masks_list = []
    control_imgs_list = []
    
    # 1. 解包
    for item in batch:
        imgs.append(item[0])
        prompt_embeds_list.append(item[1])
        prompt_masks_list.append(item[2])
        control_imgs_list.append(item[3])
    
    # 2. 堆叠 GT Images (假设它们尺寸一致，否则这里也会报错)
    # 如果你的 image_resize 逻辑正确，这里应该是 [B, C, H, W]
    gt_images = torch.stack(imgs)
    
    # 3. 动态 Padding Text Embeddings
    # 找出当前 batch 中最长的序列长度
    max_len = max([p.shape[0] for p in prompt_embeds_list])
    embed_dim = prompt_embeds_list[0].shape[1] # 3584
    
    # 初始化全零 Tensor
    padded_embeds = torch.zeros(len(batch), max_len, embed_dim, dtype=prompt_embeds_list[0].dtype)
    padded_masks = torch.zeros(len(batch), max_len, dtype=prompt_masks_list[0].dtype)
    
    # 填入数据
    for i, (emb, mask) in enumerate(zip(prompt_embeds_list, prompt_masks_list)):
        seq_len = emb.shape[0]
        padded_embeds[i, :seq_len, :] = emb
        padded_masks[i, :seq_len] = mask

    # 4. 处理 Control Image Dict (字典里的 Tensor 也需要堆叠)
    # 假设 control_img_dict 里的 'image_latents' 等也是 packed 序列，
    # 如果它们长度也不同（因为图片长宽比不同），这里也需要类似的 Pad 逻辑！
    # 假设目前 Control Image 也是变长的 (基于 Packed 格式):
    
    keys = control_imgs_list[0].keys()
    collated_control = {}
    
    for k in keys:
        # 收集该 key 下的所有 tensor
        tensors = [d[k] for d in control_imgs_list]
        
        # 如果是 Tensor
        if isinstance(tensors[0], torch.Tensor):
            # 检查是否是序列数据 (dim > 0) 且第一维是变长的
            # 对于 Packed Latents [Seq, C]，Seq 可能会变
            if tensors[0].dim() >= 1: 
                max_k_len = max([t.shape[0] for t in tensors])
                # 如果是标量或者固定维度的，直接 stack
                if max_k_len == tensors[0].shape[0] and all(t.shape[0] == max_k_len for t in tensors):
                     collated_control[k] = torch.stack(tensors)
                else:
                     # 需要 Pad (例如 Packed Latents)
                     # 注意：Packed Latents Pad 0 是否会影响 Attention 需要看 Processor 实现
                     # 通常建议 Pad 到右边，并利用 indices 区分
                     # 这里简单实现：Pad 0
                     dims = list(tensors[0].shape)
                     dims[0] = max_k_len
                     padded_k = torch.zeros(len(batch), *dims, dtype=tensors[0].dtype)
                     for i, t in enumerate(tensors):
                         padded_k[i, :t.shape[0], ...] = t
                     collated_control[k] = padded_k
            else:
                collated_control[k] = torch.stack(tensors)
        else:
            # 如果是 int/float (如 L_noise)
            collated_control[k] = torch.tensor(tensors)

    return gt_images, padded_embeds, padded_masks, collated_control


def loader(train_batch_size, num_workers, **args):
    dataset = CustomImageDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers,collate_fn=collate_fn, shuffle=True)
