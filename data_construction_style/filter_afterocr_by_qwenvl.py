import os
import json
import torch
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# === 配置 ===
INPUT_JSON = "/app/cold1/code/texteditRoPE/data_construction_style/final_training_dataset_3.json" # 上一步 OCR 筛选后的结果
OUTPUT_CLEAN_JSON = "final_clean_dataset_qwen_3.json" # 最终保留的数据
OUTPUT_REJECT_JSON = "rejected_samples_qwen_3.json"   # 被大模型剔除的数据 (用于检查)

# 显存优化配置 (建议开启 BF16)
TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"--- Initializing Qwen2.5-VL on {DEVICE} with {TORCH_DTYPE} ---")

# 1. 加载模型 (使用你提供的加载方式)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/app/cold1/Qwen2.5-VL-32B-Instruct",
    torch_dtype=TORCH_DTYPE,
    #attn_implementation="flash_attention_2", # 如果显卡支持 (如3090/4090/A100)，建议开启以加速
    device_map="auto",
)

# 加载 Processor
# min_pixels/max_pixels 可以根据你的图片分辨率 (1024x1024) 适当调整，或者保持默认
processor = AutoProcessor.from_pretrained("/app/cold1/Qwen2.5-VL-32B-Instruct")

def check_image_with_qwen(image_path, target_word):
    """
    使用 Qwen2.5-VL 检查图片质量
    返回: (bool, reason_text)
    """
    
    # 构造 Prompt：要求模型做三维度的判断
    prompt = f"""
    You are a strict data quality inspector. Look at this image containing rendered text.
    The target word is "{target_word}".
    
    Check these 3 strict criteria:
    1. **Clean Background**: The background MUST be clean white or solid color. NO random objects, NO weird artifacts, NO shadows of other objects.The image should contain only artistic text and should not contain any text outside of any objects, shapes, or unrelated color blocks.
    2. **Correct Text**: The main text clearly reads "{target_word}".
    3. **No Artifacts**: There is NO extra small text, gibberish, or floating debris around the main word.
    
    Output Format:
    If the image meets ALL criteria, output exactly: YES
    If it fails ANY criteria, output: NO
    Simply output either YES or NO, without any additional output.
    """

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path, # 支持本地路径
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # 准备推理输入
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(DEVICE)

    # 推理
    # max_new_tokens 不需太大，因为我们只需要 YES/NO
    generated_ids = model.generate(**inputs, max_new_tokens=64)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0] # 取第一条结果
    
    return output_text.strip()

def run_filtering_pipeline():
    print("--- Starting Qwen2.5-VL Data Filtering ---")
    
    if not os.path.exists(INPUT_JSON):
        print(f"Error: Input file {INPUT_JSON} not found.")
        return

    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    clean_data = []
    rejected_data = []
    
    total = len(data)
    
    for i, item in enumerate(data):
        target_path = item["target_image"]
        target_word = item["metadata"]["word"]
        
        print(f"[{i}/{total}] Checking: {target_word} ... ", end="")
        
        try:
            # 调用大模型
            result_text = check_image_with_qwen(target_path, target_word)
            
            # 判断结果
            # 模型通常会输出 "YES" 或者 "NO - reason"
            # 我们通过检查是否以 YES 开头 (忽略大小写) 来判断
            if result_text.upper().startswith("YES"):
                clean_data.append(item)
                print("PASS")
            else:
                # 记录被剔除的数据和原因
                item["reject_reason"] = result_text
                rejected_data.append(item)
                print(f"REJECT ({result_text})")
                
        except Exception as e:
            print(f"ERROR: {e}")
            # 如果推理报错（比如图片损坏），稳妥起见先记录到 rejected
            item["reject_reason"] = f"Inference Error: {e}"
            rejected_data.append(item)
        
        # === 增量保存 (每50条存一次，防止程序中断白跑) ===
        if (i + 1) % 50 == 0:
            with open(OUTPUT_CLEAN_JSON, 'w', encoding='utf-8') as f:
                json.dump(clean_data, f, indent=4, ensure_ascii=False)
            print(f"  -> Saved checkpoint: {len(clean_data)} valid items so far.")

    # === 最终保存 ===
    with open(OUTPUT_CLEAN_JSON, 'w', encoding='utf-8') as f:
        json.dump(clean_data, f, indent=4, ensure_ascii=False)
        
    with open(OUTPUT_REJECT_JSON, 'w', encoding='utf-8') as f:
        json.dump(rejected_data, f, indent=4, ensure_ascii=False)

    print("\n--- Filtering Complete ---")
    print(f"Total processed: {total}")
    print(f"Kept (Clean):    {len(clean_data)} -> {OUTPUT_CLEAN_JSON}")
    print(f"Rejected:        {len(rejected_data)} -> {OUTPUT_REJECT_JSON}")

if __name__ == "__main__":
    run_filtering_pipeline()