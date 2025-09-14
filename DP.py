import os
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np
import cv2
from tqdm import tqdm
from torchvision import transforms

folder_names = os.listdir("/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/PortraitDataset/OGData")
#styles = ['academicism','baroque','neoclassicism','orientalism','realism','Ukiyo_e']

for folder in folder_names:
    input_dir =  f'/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/PortraitDataset/OGData/{folder}'
    output_dir = f'/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/PortraitDataset/ResizedData/{folder}'
    os.makedirs(output_dir, exist_ok=True)

    target_size = (384, 384)
    normalize = False
    Image.MAX_IMAGE_PIXELS = None 
    image_resize_cond = True # Resizing images or cropping
    cna = False  # Contrast and Brightness Adjustment

##For downscaling images
    def image_resize(image, target_size=(384, 384)):
        img_copy = image.copy()
        img_resized = img_copy.resize(target_size, resample=Image.Resampling.LANCZOS)
        return img_resized

##For upscaling images
    def image_resize(image, target_size=(384, 384)):
        img_copy = image.copy()
        img_resized = img_copy.resize(target_size, resample=Image.Resampling.LANCZOS)
        return img_resized


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

    # Removing cv2 implementation
    #def remove_salt_pepper_noise(image, kernel_size=3):
    #   """Remove salt and pepper noise using median filtering"""
    #  return cv2.medianBlur(image, kernel_size)

    def remove_salt_pepper_noise_pil(image, radius=1):
        """Remove salt and pepper noise using median filter"""
        return image.filter(ImageFilter.MedianFilter(size=radius))


    def sharpen_image_pil(image):
        """Sharpen image using built-in sharpen filter"""
        # Option 1: PIL built-in sharpen filter (simple)
        return image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))


    # Process all images
    for filename in tqdm(os.listdir(input_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            original_ext = os.path.splitext(filename)[1].lower()  # e.g. .jpg, .jpeg
            output_path = os.path.join(output_dir, filename)

            try:
                with Image.open(input_path) as img:
                    img = img.convert('RGB')
                    width, height = img.size
                    if image_resize_cond:
                        img = image_resize(img, target_size) # Resize the image
                    else:
                        #Crop the image to make sure it is 384 by 384
                        transform = transforms.Compose([transforms.CenterCrop(224)])
                        img = transform(img)
                    
                    # Preprocessing
    #                img = remove_salt_pepper_noise_pil(img)
                    #img_copy_np = np.array(img)
                    
                    #Contrast and Brightness Adjustment
                    if cna:
                        img = Image.fromarray(img_copy_np) #Perform cna
                        img, alpha, beta = automatic_brightness_and_contrast(img)
                        img_copy_np = np.array(img)
                    
                    # Create the sharpening kernel
                    #kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                    #img_copy_np = cv2.filter2D(img_copy_np, -1, kernel) # Sharpen the image
                    #img = Image.fromarray(img_copy_np)
                    
                    # Sharpen the image
                    #img = sharpen_image_pil(img) Disabling as results not that good

                    # Normalize if needed
                    if normalize:
                        img_np = np.array(img)
                        img_np = normalize_images(img_np)
                        img_np = (img_np * 255).astype(np.uint8)
                        img = Image.fromarray(img_np)

                    # Save
                    img.save(output_path,quality=100,optimize=True,format=img.format) # Save the result

            except Exception as e:
                print(f"Error processing {input_path}: {e}")

    print("Image preprocessing completed.")
