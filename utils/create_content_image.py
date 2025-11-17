from PIL import Image, ImageDraw, ImageFont

def create_image(text, image_size=(1024, 1024), font_size=40):
    # 创建白色背景的图像
    image = Image.new('RGB', image_size, (255, 255, 255))
    draw = ImageDraw.Draw(image)

    # 使用指定字号的字体
    try:
        # 尝试使用系统字体
        font = ImageFont.truetype("/app/code/EasyText/font/arial.ttf", font_size)
    except IOError:
        # 如果系统字体不可用，则使用默认字体并尝试调整大小
        font = ImageFont.load_default()
        print("Default font loaded. Font size may not be as expected.")
        # 注意：对于默认字体，如果需要特定大小，可能需要安装PIL时包含字体支持
        # 或者可以指定其他可用字体路径

    # 计算文本的宽度和高度，以便将其居中
    bbox = draw.textbbox((0, 0), text, font=font)  # 获取文本的边界框
    text_width = bbox[2] - bbox[0]  # 文本的宽度
    text_height = bbox[3] - bbox[1]  # 文本的高度

    position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2)

    # 使用黑色字体绘制文本
    draw.text(position, text, font=font, fill=(0, 0, 0))

    # 保存图像
    image.save('text_image.png')


# 调用函数
create_image("jade", font_size=260)