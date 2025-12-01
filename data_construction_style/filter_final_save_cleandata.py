import os
import json
import shutil
import argparse

# === 配置 ===
# 你的源 JSON 文件路径 (Step 3 生成的最终合格列表)
SOURCE_JSON_PATH = "/app/cold1/code/texteditRoPE/data_construction_style/final_clean_dataset_qwen_3.json"

# 你希望保存清洗后数据的根目录
TARGET_ROOT_DIR = "/app/cold1/code/texteditRoPE/data_construction_style/clean_dataset_final"

def reorganize_dataset():
    print(f"--- Starting Dataset Reorganization ---")
    print(f"Source Metadata: {SOURCE_JSON_PATH}")
    print(f"Target Directory: {TARGET_ROOT_DIR}")

    # 1. 读取源元数据
    if not os.path.exists(SOURCE_JSON_PATH):
        print(f"Error: Source JSON not found: {SOURCE_JSON_PATH}")
        return

    with open(SOURCE_JSON_PATH, 'r', encoding='utf-8') as f:
        valid_data_list = json.load(f)
    
    print(f"Found {len(valid_data_list)} valid pairs to process.")

    # 2. 创建新的目录结构
    new_content_dir = os.path.join(TARGET_ROOT_DIR, "content_images")
    new_style_dir = os.path.join(TARGET_ROOT_DIR, "style_images")
    new_images_dir = os.path.join(TARGET_ROOT_DIR, "images") # 这里存放 GT 图和 txt 指令

    os.makedirs(new_content_dir, exist_ok=True)
    os.makedirs(new_style_dir, exist_ok=True)
    os.makedirs(new_images_dir, exist_ok=True)

    new_metadata = []
    appened_id = 2547  # 如果你想从某个 ID 开始编号，可以修改这里

    # 3. 遍历并处理
    # 使用 enumerate 重新生成从 0 开始的连续 ID
    for new_idx, item in enumerate(valid_data_list):
        new_idx = appened_id+new_idx
        # 获取旧的文件路径
        old_input_path = item["input_image"]
        old_style_path = item["style_image"]
        old_target_path = item["target_image"]
        
        instruction_text = item["instruction"]
        
        # 定义新的文件名 (使用 5 位数字补零，例如 pair_00001)
        # 保持后缀名一致 (虽然你之前代码都是保存为 .jpg)
        filename_base = f"{new_idx:05d}"
        
        new_input_name = f"{filename_base}.jpg"
        new_style_name = f"{filename_base}.jpg" # 或者保持 ref
        new_target_name = f"{filename_base}.jpg"
        new_txt_name = f"{filename_base}.txt"

        # 构建新的完整路径
        new_input_path = os.path.join(new_content_dir, new_input_name)
        new_style_path = os.path.join(new_style_dir, new_style_name)
        new_target_path = os.path.join(new_images_dir, new_target_name)
        new_txt_path = os.path.join(new_images_dir, new_txt_name)

        try:
            # --- 核心操作：复制文件 ---
            # 使用 shutil.copy2 可以保留文件元数据(创建时间等)
            
            # 1. 复制 Content Image (Input)
            if os.path.exists(old_input_path):
                shutil.copy2(old_input_path, new_input_path)
            else:
                print(f"Warning: Missing input file {old_input_path}, skipping pair.")
                continue

            # 2. 复制 Style Image
            if os.path.exists(old_style_path):
                shutil.copy2(old_style_path, new_style_path)
            else:
                print(f"Warning: Missing style file {old_style_path}, skipping pair.")
                continue

            # 3. 复制 Target Image (GT)
            if os.path.exists(old_target_path):
                shutil.copy2(old_target_path, new_target_path)
            else:
                print(f"Warning: Missing target file {old_target_path}, skipping pair.")
                continue

            # 4. 创建 TXT 指令文件
            with open(new_txt_path, 'w', encoding='utf-8') as f_txt:
                f_txt.write(instruction_text)

            # 5. 更新元数据
            # 记录相对路径或者绝对路径，这里建议存相对路径，方便迁移
            new_entry = {
                "id": new_idx,
                "original_id": item["id"], # 保留原始 ID 方便追溯
                "input_image": os.path.abspath(new_input_path),
                "style_image": os.path.abspath(new_style_path),
                "target_image": os.path.abspath(new_target_path),
                "instruction_file": os.path.abspath(new_txt_path),
                "instruction": instruction_text,
                "metadata": item["metadata"] # 保留之前的 OCR 框信息等
            }
            new_metadata.append(new_entry)

            if new_idx % 100 == 0:
                print(f"Processed {new_idx}/{len(valid_data_list)}...", end="\r")

        except Exception as e:
            print(f"Error processing index {new_idx} (Old ID {item['id']}): {e}")

    # 4. 保存新的总元数据
    new_json_path = os.path.join(TARGET_ROOT_DIR, "dataset.json")
    with open(new_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_metadata, f, indent=4, ensure_ascii=False)

    print(f"\n\n--- Reorganization Complete ---")
    print(f"Total pairs: {len(new_metadata)}")
    print(f"New dataset saved at: {TARGET_ROOT_DIR}")
    print(f"Instruction text files generated in: {new_images_dir}")

if __name__ == "__main__":
    reorganize_dataset()