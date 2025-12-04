import os

# === 路径配置 ===
BASE_DIR = "/app/cold1/code/texteditRoPE/data_construction_overall"  # 你的工作根目录
RAW_BG_DIR = os.path.join(BASE_DIR, "raw_backgrounds")        # 初始干净背景图文件夹
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset_output_v1")      # 输出总目录

# 资源文件
STYLE_BASE="/app/cold1/code/texteditRoPE/data_construction_style"
FONT_PATH = "/app/cold1/simhei.ttf"
STYLE_FILE = os.path.join(STYLE_BASE, "styles_corpus.json")     # 风格描述JSON
WORD_LIST = os.path.join("/app/cold1/code/texteditRoPE/data_construction_style/assets/word_list.py")               # 单词列表

# 模型路径
MODEL_QWEN_PATH = "/app/cold1/Qwen-Image-Edit-2509" 
MODEL_ZIMAGE_PATH = "/app/cold1/Z-Image-Turbo" 
RMBG_MODEL_PATH = "/app/cold1/briaai/RMBG-2.0/model.onnx" # 分割模型路径
# === 参数配置 ===
IMAGE_SIZE = (1024, 1024)
NUM_SAMPLES = 200  # 计划生成的数据量

# 自动创建子目录
SUB_DIRS = {
    "mask": os.path.join(OUTPUT_DIR, "masks"),
    "image": os.path.join(OUTPUT_DIR, "images"),     # 复制原本的背景图
    "content": os.path.join(OUTPUT_DIR, "contents"), # Step 2 结果
    "style": os.path.join(OUTPUT_DIR, "styles"),     # Step 3 结果
    "meta": os.path.join(OUTPUT_DIR, "metadata")     # 存放中间json
}

for d in SUB_DIRS.values():
    os.makedirs(d, exist_ok=True)