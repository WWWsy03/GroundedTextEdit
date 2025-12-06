"""
用于查看和保存 Poster100K 数据集的样本内容，包括图片、标题、区域坐标等信息。
"""
from datasets import load_dataset
import os
import json
from PIL import Image
from io import BytesIO

# 加载数据集
dataset = load_dataset("/app/cold1/datasets/Poster100K")

# 创建保存目录
output_dir = "/app/cold1/datasets/Poster100K/poster100kdemo"
os.makedirs(output_dir, exist_ok=True)

# 创建子目录用于分类存储
image_dir = os.path.join(output_dir, "images")
caption_dir = os.path.join(output_dir, "captions")
mask_regions_dir = os.path.join(output_dir, "mask_regions")
metadata_dir = os.path.join(output_dir, "metadata")

os.makedirs(image_dir, exist_ok=True)
os.makedirs(caption_dir, exist_ok=True)
os.makedirs(mask_regions_dir, exist_ok=True)
os.makedirs(metadata_dir, exist_ok=True)

# 获取前5个样本
for i in range(10):
    sample = dataset['train'][i]
    
    # 处理图片 - 检查是PIL Image对象还是字节数据
    image_filename = f"poster_{i:08d}.jpg"
    image_path = os.path.join(image_dir, image_filename)
    # if i<10429:  #上次存了10430张
    #     continue

    try:
        if isinstance(sample['image'], bytes):
            # 如果是字节数据，转换为PIL Image
            image = Image.open(BytesIO(sample['image']))
            image = image.convert('RGB')
            # 尝试保存图片，如果失败则跳过
            image.save(image_path)
        elif hasattr(sample['image'], 'convert'):
            # 如果是PIL Image对象
            image = sample['image'].convert('RGB')
            # 尝试保存图片，如果失败则跳过
            image.save(image_path)
        else:
            # 如果是包含路径的字节或其他格式
            print(f"Sample {i} image type: {type(sample['image'])}")
            # 尝试直接保存（如果已经是路径）
            if hasattr(sample['image'], 'save'):
                image = sample['image'].convert('RGB')
                image.save(image_path)
            else:
                print(f"无法处理Sample {i}的图片格式: {type(sample['image'])}")
                continue  # 跳过当前样本
    except Exception as e:
        print(f"Sample {i}: 保存图片失败，错误信息: {str(e)}，跳过此样本")
        continue  # 跳过当前样本
    
    # 保存caption到文本文件
    caption_filename = f"poster_{i:08d}_caption.txt"
    caption_path = os.path.join(caption_dir, caption_filename)
    with open(caption_path, 'w', encoding='utf-8') as f:
        f.write(sample['caption'])
    
    # 保存mask regions到JSON文件
    mask_filename = f"poster_{i:08d}_mask_regions.json"
    mask_path = os.path.join(mask_regions_dir, mask_filename)
    with open(mask_path, 'w', encoding='utf-8') as f:
        json.dump(sample['mask_regions'], f, indent=2, ensure_ascii=False)
    
    # 保存完整的metadata到文本文件
    metadata_filename = f"poster_{i:08d}_metadata.txt"
    metadata_path = os.path.join(metadata_dir, metadata_filename)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + f"\nSAMPLE {i}\n" + "=" * 50 + "\n")
        f.write(f"File Name: {sample['file_name']}\n")
        f.write(f"Folder Path: {sample['folder_path']}\n")
        f.write(f"Batch Name: {sample['batch_name']}\n")
        f.write(f"Normalized Path: {sample['normalized_path']}\n")
        f.write(f"Number of mask regions: {len(sample['mask_regions']) if isinstance(sample['mask_regions'], list) else 'N/A'}\n")
        f.write("-" * 30 + "\n")
        f.write(f"Caption:\n{sample['caption']}\n")
        f.write("-" * 30 + "\n")
        f.write(f"Mask Regions:\n{json.dumps(sample['mask_regions'], indent=2, ensure_ascii=False)}\n")
    
    print(f"Sample {i}:")
    print(f"  - Image: {sample['file_name']}")
    print(f"  - Batch: {sample['batch_name']}")
    print(f"  - Caption length: {len(sample['caption'])} chars")
    print(f"  - Mask regions count: {len(sample['mask_regions']) if isinstance(sample['mask_regions'], list) else 'N/A'}")
    print(f"  - Caption preview: {sample['caption'][:80]}...")
    print("-" * 80)

print(f"\n所有文件已保存到目录: {output_dir}")
print(f"包含以下子目录:")
print(f"  - images/: 保存图片文件")
print(f"  - captions/: 保存标题文本")
print(f"  - mask_regions/: 保存区域坐标JSON文件") 
print(f"  - metadata/: 保存完整元数据")

# 统计信息
print(f"\n数据集统计:")
print(f"总样本数: {len(dataset['train'])}")
