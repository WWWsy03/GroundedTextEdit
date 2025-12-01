import torch
import numpy as np
import json
import os
# 1. 加载 tensor

#*******读gt 纯tensor*********
# tensor = torch.load('/app/cold1/code/texteditRoPE/qwenimage-style-control-double-output/cache/img_embs/img1.pt', map_location='cpu')  # 确保在 CPU 上加载
# tensor_f32 = tensor.float()  # 等价于 .to(torch.float32)

# # 3. 转 numpy
# arr = tensor_f32.detach().cpu().numpy()

# # 3. 保存为文本（支持多维）
# import numpy as np
# np.savetxt('gt_tensor.txt', arr.reshape(-1, arr.shape[-1]) if arr.ndim > 1 else arr, fmt='%.6f')

data = torch.load('/app/cold1/code/texteditRoPE/qwenimage-style-control-double-output/cache/img_embs_control/img1.pt', map_location='cpu')

output_dir = 'control_dump'
os.makedirs(output_dir, exist_ok=True)

def convert_item(key, value, idx=0):
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu()
        if arr.dtype == torch.bfloat16:
            arr = arr.float()  # 兼容 numpy
        arr=arr.detach().cpu().numpy()
        npy_path = f"{output_dir}/{key}_{idx}.txt"
        np.savetxt(npy_path, arr.reshape(-1, arr.shape[-1]) if arr.ndim > 1 else arr, fmt='%.6f')
        #np.save(npy_path, arr.numpy())
        return {
            "type": "tensor",
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "saved_as": os.path.basename(npy_path)
        }
    elif isinstance(value, np.ndarray):
        npy_path = f"{output_dir}/{key}_{idx}.txt"
        np.save(npy_path, value)
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "saved_as": os.path.basename(npy_path)
        }
    elif isinstance(value, dict):
        return {k: convert_item(k, v, i) for i, (k, v) in enumerate(value.items())}
    elif isinstance(value, list):
        return [convert_item(f"{key}_list", v, i) for i, v in enumerate(value)]
    else:
        return value  # 基本类型直接保留

converted = {}
for i, (k, v) in enumerate(data.items()):
    converted[k] = convert_item(k, v, i)

# 保存元信息
with open(f'{output_dir}/meta.json', 'w', encoding='utf-8') as f:
    json.dump(converted, f, indent=2, ensure_ascii=False)