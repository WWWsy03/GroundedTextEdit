import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
import os

# === 配置 ===
MODEL_PATH = "/app/cold1/Qwen/Qwen2.5-7B-Instruct" 
OUTPUT_STYLE_FILE = "/app/cold1/code/texteditRoPE/data_construction_style/styles_corpus.json"
TOTAL_STYLES = 4000   # 总共需要生成的数量
BATCH_SIZE = 10      # 每一轮对话生成的数量

# === 极简且严格的系统提示词 ===
# 使用 Struct format 指令，减少废话
SYSTEM_PROMPT = """
You are a backend data generator API. You have NO conversational capability. 
Your ONLY function is to return a raw JSON list containing {batch_size} text style descriptions.

**Task Requirements:**
1. Generate highly imaginative style descriptions for 3D text rendering.
2. Elements to combine: Material(e.g., liquid gold, fur, slime, neon glass, rusting iron, holographic data), Font Shape(e.g., bubble, gothic, jagged, pixelated, ribbons), Lighting(e.g., cinematic, dark, glowing, studio softbox), Color.
3. Constraint: Describe the TEXT object only. Do not describe a complex background environment.The description should only refer to the font style and cannot include descriptions of the background.
The examples are for reference only; please feel free to use your imagination to generate diverse results.

**Strict Output Format Rules:**
- The output must start immediately with `[` and end with `]`.
- NO introductory text (e.g., "Here is the list", "Sure", "I thought").
- NO Markdown code blocks (e.g., no ```json).
- NO whitespace or newlines before the opening bracket.
- Output MUST be valid JSON.
直接给我JSON直接给我JSON不要回答其他内容！！！！

**Example of Exact Expected Output:**
["Translucent blue jelly text with internal bubbles", "Rusty iron block letters with oil stains", "Neon glowing cybernetic font with circuit patterns", "Fluffy white fur text in soft studio lighting"]
The examples are for reference only; please feel free to use your imagination to generate diverse results.
"""

def extract_json_from_text(text):
    """
    辅助函数：即使模型输出了少量废话，也通过正则强制提取 JSON 列表
    """
    try:
        # 1. 尝试直接解析（如果模型非常听话）
        return json.loads(text)
    except:
        # 2. 如果包含 Markdown 或废话，使用正则提取最外层的 []
        try:
            matches = re.findall(r'\[.*\]', text, re.DOTALL)
            if matches:
                # 找到最长的一个匹配项（防止匹配到内部的小括号）
                json_str = max(matches, key=len)
                return json.loads(json_str)
        except:
            pass
    return []

def generate_styles_iterative():
    print(f"--- Starting Generation: Goal {TOTAL_STYLES} styles ---")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="auto"
    )

    all_styles = []
    
    # 断点续传逻辑
    if os.path.exists(OUTPUT_STYLE_FILE):
        try:
            with open(OUTPUT_STYLE_FILE, 'r', encoding='utf-8') as f:
                all_styles = json.load(f)
            print(f"Resuming from existing file. Found {len(all_styles)} styles.")
        except:
            all_styles = []

    # 初始化第一轮对话
    # 注意：SYSTEM_PROMPT 需要先 format 填入 batch_size
    sys_content = SYSTEM_PROMPT.format(batch_size=BATCH_SIZE)
    messages = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": f"Generate the first batch of {BATCH_SIZE} unique text styles."}
    ]

    while len(all_styles) < TOTAL_STYLES:
        # 准备输入
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        print(f"Generating batch... ({len(all_styles)}/{TOTAL_STYLES})")
        
        # 生成
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 解析数据
        new_batch = extract_json_from_text(response_text)
        
        if new_batch and isinstance(new_batch, list):
            print(f"  -> Valid JSON received: {len(new_batch)} items.")
            
            # 存入总列表（简单去重）
            for style in new_batch:
                print(f"     - {style}")
                if isinstance(style, str) and style not in all_styles:
                    all_styles.append(style)
            
            # 实时写回文件
            with open(OUTPUT_STYLE_FILE, 'w', encoding='utf-8') as f:
                json.dump(all_styles, f, indent=4)

            # === 关键修改：多轮对话逻辑 ===
            # 1. 将助手的回答加入历史，这样模型就知道它刚才生成了什么
            messages.append({"role": "assistant", "content": response_text})
            
            # 2. 添加下一轮的用户指令，明确要求“不重复”
            messages.append({
                "role": "user", 
                "content": f"Generate {BATCH_SIZE} MORE styles. Use completely different materials and colors from the ones above. Strict JSON format.No other output."
            })
            
            # 3. 历史记录清理（滑动窗口）
            # 如果对话轮次超过 3 轮（3次 User + 3次 Assistant + 1次 System = 7条消息），清理旧历史
            # 这样可以防止上下文过长导致 OOM 或 混乱，同时保留最近的上下文
            if len(messages) > 7:
                print("  -> Clearing context window to maintain freshness...")
                messages = [
                    {"role": "system", "content": sys_content},
                    # 假装这是新的请求，但带有“不同”的暗示
                    {"role": "user", "content": f"Generate {BATCH_SIZE} new text styles, distinct from typical ones. Focus on exotic textures."}
                ]
                
        else:
            print(f"  -> Error parsing JSON. Raw output: {response_text[:50]}...")
            # 如果解析失败，不要更新 messages，直接重试这一轮
            continue

    print(f"--- Done! Total {len(all_styles)} styles saved to {OUTPUT_STYLE_FILE} ---")
    return all_styles

if __name__ == "__main__":
    generate_styles_iterative()