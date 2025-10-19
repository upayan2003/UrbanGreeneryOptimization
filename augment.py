import os
import glob
import cv2
import numpy as np
from scipy import ndimage

# --- Configuration ---
# This should be the 'train' folder created by the split_dataset.py script.
INPUT_DIR = os.path.join('greenery_dataset', 'train')

# A new folder where the augmented dataset will be saved.
OUTPUT_DIR = 'augmented_train' 

# --- Augmentation Parameters ---
ROTATION_ANGLES = [90, 180, 270]
BRIGHTNESS_RANGE = (-0.1, 0.1) # Add/subtract this fraction of the max value
CONTRAST_RANGE = (0.9, 1.1) # Multiply by a factor in this range

def augment_dataset(input_dir, output_dir):
    """
    Applies augmentations using SciPy and NumPy, which are robust for any
    TIFF data type (integer or float).
    """
    input_image_dir = os.path.join(input_dir, 'images')
    input_mask_dir = os.path.join(input_dir, 'masks')
    
    output_image_dir = os.path.join(output_dir, 'images')
    output_mask_dir = os.path.join(output_dir, 'masks')

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    image_paths = glob.glob(os.path.join(input_image_dir, '*.tif'))
    total_files = len(image_paths)
    print(f"Found {total_files} images to augment in '{input_image_dir}'.")
    
    for i, image_path in enumerate(image_paths):
        base_filename = os.path.basename(image_path)
        mask_filename = base_filename.replace('.tif', '_mask.tif')
        mask_path = os.path.join(input_mask_dir, mask_filename)

        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if image is None or mask is None:
            print(f"Warning: Could not read image or mask for {base_filename}. Skipping.")
            continue
        
        print(f"Processing ({i+1}/{total_files}): {base_filename}")
        
        # --- A. Save the original files first ---
        cv2.imwrite(os.path.join(output_image_dir, base_filename), image)
        cv2.imwrite(os.path.join(output_mask_dir, mask_filename), mask)

        # --- B. Geometric Augmentations (apply to both image and mask) ---
        
        # Rotations using SciPy
        for angle in ROTATION_ANGLES:
            # reshape=False prevents SciPy from changing the image dimensions
            rotated_image = ndimage.rotate(image, angle, reshape=False, order=1)
            rotated_mask = ndimage.rotate(mask, angle, reshape=False, order=0) # order=0 for nearest neighbor on masks
            
            cv2.imwrite(os.path.join(output_image_dir, f"{base_filename.split('.')[0]}_rot{angle}.tif"), rotated_image)
            cv2.imwrite(os.path.join(output_mask_dir, f"{mask_filename.split('_mask.')[0]}_rot{angle}_mask.tif"), rotated_mask)
            
        # Flipping using NumPy
        h_flipped_image = np.fliplr(image)
        h_flipped_mask = np.fliplr(mask)
        cv2.imwrite(os.path.join(output_image_dir, f"{base_filename.split('.')[0]}_hflip.tif"), h_flipped_image)
        cv2.imwrite(os.path.join(output_mask_dir, f"{mask_filename.split('_mask.')[0]}_hflip_mask.tif"), h_flipped_mask)

        v_flipped_image = np.flipud(image)
        v_flipped_mask = np.flipud(mask)
        cv2.imwrite(os.path.join(output_image_dir, f"{base_filename.split('.')[0]}_vflip.tif"), v_flipped_image)
        cv2.imwrite(os.path.join(output_mask_dir, f"{mask_filename.split('_mask.')[0]}_vflip_mask.tif"), v_flipped_mask)
        
        # --- C. Photometric/Color Augmentations (apply ONLY to the image) ---
        bgr = image[:, :, :3]
        nir = image[:, :, 3]
        
        is_float = image.dtype.kind == 'f'
        dtype_max = 1.0 if is_float else np.iinfo(image.dtype).max

        # Brightness
        brightness_factor = np.random.uniform(BRIGHTNESS_RANGE[0], BRIGHTNESS_RANGE[1])
        bright_bgr = bgr + brightness_factor * dtype_max
        
        # Contrast
        contrast_factor = np.random.uniform(CONTRAST_RANGE[0], CONTRAST_RANGE[1])
        # We perform contrast around the mean pixel value to make it more natural
        mean_val = np.mean(bgr)
        contrast_bgr = (bgr - mean_val) * contrast_factor + mean_val

        # Clip values to ensure they are in the valid range for the data type
        bright_bgr = np.clip(bright_bgr, 0, dtype_max)
        contrast_bgr = np.clip(contrast_bgr, 0, dtype_max)

        # Recombine with NIR and save
        bright_image = np.dstack((bright_bgr.astype(image.dtype), nir))
        contrast_image = np.dstack((contrast_bgr.astype(image.dtype), nir))
        
        cv2.imwrite(os.path.join(output_image_dir, f"{base_filename.split('.')[0]}_bright.tif"), bright_image)
        cv2.imwrite(os.path.join(output_mask_dir, f"{mask_filename.split('_mask.')[0]}_bright_mask.tif"), mask)
        
        cv2.imwrite(os.path.join(output_image_dir, f"{base_filename.split('.')[0]}_contrast.tif"), contrast_image)
        cv2.imwrite(os.path.join(output_mask_dir, f"{mask_filename.split('_mask.')[0]}_contrast_mask.tif"), mask)


    print("\nData augmentation complete!")
    print(f"Augmented training set saved in '{output_dir}'.")
    final_count = len(glob.glob(os.path.join(output_image_dir, '*.tif')))
    print(f"Total files after augmentation: {final_count}")


if __name__ == '__main__':

    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory not found at '{INPUT_DIR}'.")
        print("Please make sure you have run the 'split_dataset.py' script and the 'train' folder exists.")
    else:
        augment_dataset(INPUT_DIR, OUTPUT_DIR)