import os
import glob
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torchvision.transforms.functional as TF

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# -------------------- Configuration --------------------
DATA_DIR = 'greenery_dataset'
AUGMENTED_DIR = 'augmented_train'
MODEL_SAVE_PATH = "unet_greenery_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_EPOCHS = 15
LEARNING_RATE = 1e-5
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
NUM_WORKERS = 2

# Calculated dataset-specific values
DATASET_MEAN = [755.1042, 845.9725, 901.3823, 1868.3954]
DATASET_STD = [852.1098, 826.0251, 946.7518, 1090.7274]
POS_WEIGHT = 3.4785

# -------------------- U-Net Model (with shape fix) --------------------
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
        x = TF.resize(x, size=x4.shape[2:])
        x = torch.cat([x, x4], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = TF.resize(x, size=x3.shape[2:])
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = TF.resize(x, size=x2.shape[2:])
        x = torch.cat([x, x2], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        x = TF.resize(x, size=x1.shape[2:])
        x = torch.cat([x, x1], dim=1)
        x = self.conv4(x)
        
        return self.outc(x)

# -------------------- Custom Dataset --------------------
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

# -------------------- Loss Function --------------------
class DiceBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super(DiceBCEWithLogitsLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, inputs, targets, smooth=1):
        targets = targets.float()
        bce = self.bce_loss(inputs, targets)
        
        inputs_prob = torch.sigmoid(inputs)
        inputs_flat = inputs_prob.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)
        
        return bce + dice_loss

# -------------------- Dataloaders & Transforms --------------------
def get_dataloaders(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir, batch_size):
    train_transform = A.Compose([
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # <--- FIX: Removed max_pixel_value for correct float normalization
        A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
        # <--- FIX: Removed max_pixel_value
        A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
        ToTensorV2(),
    ])

    train_dataset = GreeneryDataset(train_img_dir, train_mask_dir, transform=train_transform)
    val_dataset = GreeneryDataset(val_img_dir, val_mask_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    return train_loader, val_loader

# -------------------- Training & Evaluation --------------------
def check_accuracy(loader, model, device=DEVICE):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1) # Add channel dim
            
            # <--- FIX: Changed logic from argmax to sigmoid + threshold for single-channel output
            logits = model(x)
            preds = (torch.sigmoid(logits) > 0.5).float()
            
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    accuracy = num_correct / num_pixels
    avg_dice = dice_score / len(loader)
    
    print(f"\nValidation Accuracy: {accuracy*100:.2f}%")
    print(f"Dice Score: {avg_dice:.4f}")
    
    model.train()
    return accuracy, avg_dice

def train_model(model, train_loader, val_loader, device):
    pos_weight_tensor = torch.tensor([POS_WEIGHT], device=device)
    criterion = DiceBCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {'train_loss': [], 'val_acc': [], 'val_dice': []}
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device).unsqueeze(1).float()
            
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        
        val_acc, val_dice = check_accuracy(val_loader, model, device=device)
        history['val_acc'].append(val_acc.item())
        history['val_dice'].append(val_dice.item())

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    return history

# -------------------- Prediction & Visualization --------------------
def predict_and_visualize(model, img_path, device, save_path="prediction.png"):
    model.eval()
    
    image = np.nan_to_num(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), nan=0.0)
    
    vis_image = image[:, :, :3]
    vis_image = (vis_image - vis_image.min()) / (vis_image.max() - vis_image.min())
    original_image_vis = (vis_image * 255).astype(np.uint8)


    transform = A.Compose([
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
        A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
        ToTensorV2(),
    ])
    
    augmented = transform(image=image)
    input_tensor = augmented['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(input_tensor)
        preds = (torch.sigmoid(logits) > 0.5).squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
        
    overlay = np.zeros_like(original_image_vis, dtype=np.uint8)
    overlay[preds == 1] = [0, 255, 0]
    
    overlay_resized = cv2.resize(overlay, (original_image_vis.shape[1], original_image_vis.shape[0]), interpolation=cv2.INTER_NEAREST)
    blended = cv2.addWeighted(original_image_vis, 1, overlay_resized, 0.6, 0)
    
    cv2.imwrite(save_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    print(f"Prediction saved to {save_path}")

def plot_and_save_graphs(history, save_path="training_plots.png"):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    plt.title('Training Loss'); plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, [acc * 100 for acc in history['val_acc']], 'ro-', label='Validation Accuracy (%)')
    plt.plot(epochs, history['val_dice'], 'go-', label='Validation Dice Score')
    
    plt.title('Validation Metrics'); plt.xlabel('Epochs'); plt.ylabel('Score')
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close() # Close plot to free memory
    print(f"Training plots saved to {save_path}")

# -------------------- Main Execution --------------------
if __name__ == '__main__':
    # Clean up CUDA cache before starting
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print(f"Using device: {DEVICE}")
    model = UNet(in_channels=4, out_channels=1).to(DEVICE)
    
    train_img_dir = os.path.join(AUGMENTED_DIR, 'images')
    train_mask_dir = os.path.join(AUGMENTED_DIR, 'masks')
    val_img_dir = os.path.join(DATA_DIR, 'val', 'images')
    val_mask_dir = os.path.join(DATA_DIR, 'val', 'masks')
    
    train_loader, val_loader = get_dataloaders(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir, BATCH_SIZE)
    
    training_history = train_model(model, train_loader, val_loader, DEVICE)
    
    plot_and_save_graphs(training_history)
    
    print("\n--- Running Prediction ---")
    # Load the best model for prediction
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    
    test_image_paths = glob.glob(os.path.join(DATA_DIR, 'test', 'images', '*.tif'))
    if test_image_paths:
        predict_and_visualize(model, test_image_paths[0], device=DEVICE)
    else:
        print("No test images found to run prediction.")