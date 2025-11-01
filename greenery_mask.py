from u_net_model import UNet, predict_and_visualize
import os
import glob
import numpy as np
import cv2
import torch
from albumentations.pytorch import ToTensorV2
import albumentations as A

# -------------------- Configuration --------------------
MODEL_SAVE_PATH = "unet_greenery_model.pth"
TARGET_DIR = "greenery_dataset/test/images" # Replace with your test images directory
OUTPUT_DIR = "predictions"

IMAGE_HEIGHT, IMAGE_WIDTH = 256, 256
DATASET_MEAN = [755.1042, 845.9725, 901.3823, 1868.3954]
DATASET_STD = [852.1098, 826.0251, 946.7518, 1090.7274]
MAX_PIXEL = 65535

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Setup --------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Using device: {DEVICE}")
print("Loading model...")

model = UNet(in_channels=4, out_channels=1).to(DEVICE)
state_dict = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

# -------------------- Prediction Loop --------------------
image_paths = sorted(glob.glob(os.path.join(TARGET_DIR, "*.tif")))

if not image_paths:
    print("No target images found to run prediction.")
else:
    print(f"Found {len(image_paths)} images in {TARGET_DIR}")
    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        overlay_path = os.path.join(OUTPUT_DIR, f"{base_name}_overlay.png")
        mask_path = os.path.join(OUTPUT_DIR, f"{base_name}_mask.tif")

        print(f"Processing: {base_name}")

        model.eval()
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Warning: Could not read {img_path}, skipping.")
            continue

        transform = A.Compose([
            A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
            A.Normalize(mean=DATASET_MEAN, std=DATASET_STD, max_pixel_value=MAX_PIXEL),
            ToTensorV2(),
        ])

        augmented = transform(image=np.nan_to_num(image, nan=0.0))
        input_tensor = augmented["image"].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(input_tensor)
            preds = (torch.sigmoid(logits) > 0.5).squeeze().cpu().numpy().astype("uint8")

        preds_resized = cv2.resize(
            preds, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
        )

        cv2.imwrite(mask_path, preds_resized * 255)
        predict_and_visualize(model, img_path, device=DEVICE, save_path=overlay_path)

    print(f"\nAll predictions saved in '{OUTPUT_DIR}' folder.")

