import os
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

# Define the input and output directories
input_dir =  'dataset/academicism_filtered_dataset'
output_dir = 'dataset/academicism_preprocessed_dataset'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define preprocessing parameters
target_size = (384, 384)  # Resize images to 224x224
normalize = True          # Normalize pixel values to [0, 1]


def image_resize(img, target_size):
    """Resize the image to the target size."""
    return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

def normalize_images(image, normalize=True):
    """Normalize the image pixel values."""
    img_array = np.array(image)
    return img_array / 255.0

# Automatic brightness and contrast optimization with optional histogram clipping
def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    if len(image.shape) == 3 and image.shape[2] >= 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)
    
    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))
    
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0
    
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    
    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    
    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    
    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    Citation: https://stackoverflow.com/questions/56905592/automatic-contrast-and-brightness-adjustment-of-a-color-photo-of-a-sheet-of-pape
    '''

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)

def remove_salt_pepper_noise(image,kernel_size=3):
    filtered_image = cv2.medianBlur(image, kernel_size)
    'Remove salt and pepper noise (electrical noise) using median filtering'
    return filtered_image

# Iterate through all images in the input directory
for filename in tqdm(os.listdir(input_dir)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        output_filename = os.path.splitext(filename)[0] + '.png'
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, output_filename)
        try:
            # Open the image
            with Image.open(input_path) as img:
                # Resize the image
                img = image_resize(np.array(img), target_size)
                
                # Remove salt and pepper noise
                img = remove_salt_pepper_noise(img)
                
                # Apply automatic brightness and contrast
                img, alpha, beta = automatic_brightness_and_contrast(img)
                
                # Normalize the image if required
                if normalize:
                    img = normalize_images(img)
                
                # Save the preprocessed image
                preprocessed_img = Image.fromarray((img * 255).astype(np.uint8))
                preprocessed_img.save(output_path)
        
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
        

print("Image preprocessing completed.")