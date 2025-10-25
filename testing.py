import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# -------------------- Configuration --------------------
# --- Paths ---
DATA_DIR = 'greenery_dataset'
MODEL_PATH = "unet_greenery_model.pth"
PREDICTIONS_FOLDER = "test_predictions/"

# --- Model & Data Parameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4  # You can use a larger batch size for testing
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
NUM_WORKERS = 2

# --- Dataset Statistics (must be the same as used in training) ---
DATASET_MEAN = [755.1042, 845.9725, 901.3823, 1868.3954]
DATASET_STD = [852.1098, 826.0251, 946.7518, 1090.7274]


# -------------------- U-Net Model & Dataset Class --------------------
# NOTE: The UNet and GreeneryDataset classes need to be defined here
# so this script can run independently.

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super(UNet, self).__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv3(x)
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv4(x)
        return self.outc(x)

class GreeneryDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.tif")))
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        base_filename = os.path.basename(img_path)
        mask_filename = f"{os.path.splitext(base_filename)[0]}_mask.tif"
        mask_path = os.path.join(self.mask_dir, mask_filename)
        
        image = np.nan_to_num(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), nan=0.0)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise FileNotFoundError(f"Mask not found for image: {img_path}")
        
        mask[mask > 0] = 1

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask

# -------------------- Corrected Dataloader Function --------------------
def get_test_dataloader(test_img_dir, test_mask_dir, batch_size):
    """Creates a DataLoader for the test set."""
    test_transform = A.Compose([
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
        A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
        ToTensorV2(),
    ])

    # **CORRECTION HERE:** Create the GreeneryDataset instance
    test_dataset = GreeneryDataset(
        image_dir=test_img_dir,
        mask_dir=test_mask_dir,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset, # Pass the created dataset object
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    return test_loader


# -------------------- Evaluation Function --------------------
def evaluate_and_save_predictions(loader, model, folder, device):
    """
    Evaluates the model on the test set, prints metrics,
    and saves visual predictions.
    """
    print("--- Running Evaluation & Saving Predictions ---")
    if not os.path.exists(folder):
        os.makedirs(folder)

    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(loader, desc="Testing")):
            x = x.to(device)
            y_gpu = y.to(device).unsqueeze(1)
            
            preds = torch.sigmoid(model(x))
            preds_binary = (preds > 0.5).float()

            num_correct += (preds_binary == y_gpu).sum()
            num_pixels += torch.numel(preds_binary)
            dice_score += (2 * (preds_binary * y_gpu).sum()) / (
                (preds_binary + y_gpu).sum() + 1e-8
            )

            # --- Save visual output ---
            x_rgb = x[:, :3, :, :].permute(0, 2, 3, 1).cpu().numpy()
            x_rgb = (x_rgb * np.array(DATASET_STD[:3])) + np.array(DATASET_MEAN[:3])
            x_rgb = np.clip(x_rgb, 0, 255).astype(np.uint8)

            preds_binary_np = preds_binary.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8) * 255
            true_mask_np = y.unsqueeze(1).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8) * 255

            for j in range(x_rgb.shape[0]):
                true_mask_3ch = cv2.cvtColor(true_mask_np[j], cv2.COLOR_GRAY2RGB)
                
                overlay = x_rgb[j].copy()
                overlay[preds_binary_np[j, :, :, 0] == 255] = [0, 255, 0]
                blended = cv2.addWeighted(x_rgb[j], 0.7, overlay, 0.3, 0)
                
                combined_image = np.concatenate([x_rgb[j], true_mask_3ch, blended], axis=1)
                
                img_index = i * loader.batch_size + j
                cv2.imwrite(os.path.join(folder, f"prediction_{img_index}.png"), cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

    final_accuracy = num_correct / num_pixels
    final_dice = dice_score / len(loader)

    print(f"\n--- Test Results ---")
    print(f"Pixel Accuracy: {final_accuracy*100:.2f}%")
    print(f"Dice Score: {final_dice:.4f}")
    print(f"Predictions saved to '{folder}'")


# -------------------- Main Execution --------------------
if __name__ == '__main__':
    test_img_dir = os.path.join(DATA_DIR, 'test', 'images')
    test_mask_dir = os.path.join(DATA_DIR, 'test', 'masks')

    test_loader = get_test_dataloader(test_img_dir, test_mask_dir, BATCH_SIZE)

    model = UNet(in_channels=4, out_channels=1).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)))
    except FileNotFoundError:
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        exit()
    
    print(f"Model loaded successfully from '{MODEL_PATH}'")
    
    evaluate_and_save_predictions(test_loader, model, PREDICTIONS_FOLDER, DEVICE)