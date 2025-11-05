from PIL import Image, ImageDraw
import numpy as np

def generate_mask_image(width, height, bbox_list, output_path):
    """
    Generates a binary mask image based on specified dimensions and bounding boxes.

    Args:
        width (int): The width of the mask image.
        height (int): The height of the mask image.
        bbox_list (list): A list of bounding boxes. Each box is a list [x_min, y_min, x_max, y_max].
        output_path (str): The file path where the mask image will be saved.
    """
    # Create a new black image (0s) with the specified dimensions in 'L' (grayscale) mode
    mask_image = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask_image)

    for bbox in bbox_list:
        # Ensure bounding box coordinates are integers
        x_min, y_min, x_max, y_max = map(int, bbox)
        # Draw a white rectangle (255s) for each bounding box on the mask
        # The bbox format for ImageDraw.rectangle is [x_min, y_min, x_max, y_max]
        draw.rectangle([x_min, y_min, x_max, y_max], fill=255)

    # Save the generated mask image to the specified path
    mask_image.save(output_path)
    print(f"Mask image saved to: {output_path}")

# Example usage:
if __name__ == "__main__":
    # Define the resolution of the mask image
    image_width = 900
    image_height = 1350

    # Define the list of bounding boxes
    bboxes = [
        [275, 930, 360, 953],
        [208, 953, 709, 1075],
        [244, 1054, 657, 1181]
    ]

    # Define the output path for the mask image
    save_path = "generated_mask.png"

    # Generate and save the mask image
    generate_mask_image(image_width, image_height, bboxes, save_path)

    # Optional: Load and display the image to verify
    # loaded_mask = Image.open(save_path)
    # loaded_mask.show() # This might open an image viewer