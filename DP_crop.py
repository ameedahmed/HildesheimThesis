import os
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np
import cv2
from tqdm import tqdm
from torchvision import transforms

normalize = False
cna = False
Image.MAX_IMAGE_PIXELS = None 

styles = ['academicism','baroque','neoclassicism','orientalism','realism','Ukiyo_e']

for style in styles:
    folders = ['train','test']
    for folder in folders:
        input_dir = f'dataset/original_{folder}_{style}_filtered_dataset'
        output_dir = f'cropped_dataset/cropped_{folder}_{style}_dataset'
        os.makedirs(output_dir, exist_ok=True)

        crop_size = 384

        def normalize_images(image):
            return image / 255.0

        def automatic_brightness_and_contrast(image, clip_hist_percent=1):
            gray = ImageOps.grayscale(image)
            gray_np = np.asarray(gray)

            hist = np.histogram(gray_np.flatten(), bins=256, range=(0,256))[0]
            hist_size = len(hist)

            accumulator = np.cumsum(hist).astype(float)

            maximum = accumulator[-1]
            clip_hist_percent *= (maximum / 100.0)
            clip_hist_percent /= 2.0

            minimum_gray = np.searchsorted(accumulator, clip_hist_percent)
            maximum_gray = np.searchsorted(accumulator, maximum - clip_hist_percent) - 1

            alpha = 255.0 / (maximum_gray - minimum_gray) if (maximum_gray - minimum_gray) != 0 else 1.0
            beta = -minimum_gray * alpha

            img_np = np.asarray(image).astype(np.float32)

            img_np = img_np * alpha + beta
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)

            auto_result = Image.fromarray(img_np)
            return auto_result, alpha, beta

        def remove_salt_pepper_noise_pil(image, radius=1):
            """Remove salt and pepper noise using median filter"""
            return image.filter(ImageFilter.MedianFilter(size=radius))

        def remove_salt_pepper_noise(image, kernel_size=3):
            """Remove salt and pepper noise using median filtering"""
            return cv2.medianBlur(image, kernel_size)

        # Process all images
        for filename in tqdm(os.listdir(input_dir)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(input_dir, filename)
                original_ext = os.path.splitext(filename)[1].lower()  # e.g. .jpg, .jpeg
                base_name = os.path.splitext(filename)[0]

                try:
                    with Image.open(input_path) as img:
                        width,height = img.size
                        # Sliding window: move 384x384 crop window across the image
                        for top in range(0, height - crop_size + 1, crop_size):
                            for left in range(0, width - crop_size + 1, crop_size):
                                crop = img.crop((left, top, left + crop_size, top + crop_size))

                                img_crop_np = np.array(crop)

                                # Preprocessing
                                #img_crop_np = remove_salt_pepper_noise_pil(img_crop_np)
                                
                                if cna:
                                    img = Image.fromarray(img_crop_np)
                                    img, alpha, beta = automatic_brightness_and_contrast(img)
                                    img_crop_np = np.array(img)

                                # Create sharpening kernel
                                #kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                                
                                #img_crop_np = cv2.filter2D(img_crop_np, -1, kernel)

                                # Optional brightness/contrast
                                # img_crop_np, alpha, beta = automatic_brightness_and_contrast(img_crop_np)

                                if normalize:
                                    img_crop_np = normalize_images(img_crop_np)
                                    img_crop_np = (img_crop_np * 255).astype(np.uint8)
                                    img_crop = Image.fromarray(img_crop_np)

                                # Save with unique filename based on crop position
                                output_filename = f"{base_name}_x{left}_y{top}{original_ext}"
                                output_path = os.path.join(output_dir, output_filename)
                                crop.save(output_path, quality=100, optimize=True, format=img.format)

                except Exception as e:
                    print(f"Error processing {input_path}: {e}")

        print(f"Image {style} {folder} preprocessing completed.")