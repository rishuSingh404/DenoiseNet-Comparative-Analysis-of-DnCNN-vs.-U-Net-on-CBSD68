#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"


# In[3]:


import os
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# Import Weights & Biases
import wandb

################################################################################
#                               DATASET
################################################################################

class CBSD68PatchDataset(Dataset):
    """
    Custom Dataset for the CBSD68 dataset that extracts patches with a given
    patch_size and stride for training/validation. For testing, it can return
    the entire resized image.

    Folder structure:
        root_dir/
           ├── noisy35/        --> Noisy images
           └── original_png/   --> Clean (ground truth) images
    """
    def __init__(
        self,
        root_dir,
        patch_size=128,
        stride=64,
        train=True,
        transform=None,
        full_image=False,
        resize=(256, 256)
    ):
        """
        Args:
            root_dir (str): Root directory of the dataset.
            patch_size (int): Size of the patches to extract.
            stride (int): Stride for patch extraction.
            train (bool): Whether this dataset is for training/validation.
            transform (callable): Optional transform to be applied on a sample.
            full_image (bool): If True, return the full resized image instead of patches.
            resize (tuple): Size to resize the original images to before patch extraction.
        """
        self.root_dir = root_dir
        self.noisy_dir = os.path.join(root_dir, 'noisy35')
        self.original_dir = os.path.join(root_dir, 'original_png')
        self.noisy_images = sorted([
            f for f in os.listdir(self.noisy_dir)
            if os.path.isfile(os.path.join(self.noisy_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.original_images = sorted([
            f for f in os.listdir(self.original_dir)
            if os.path.isfile(os.path.join(self.original_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride
        self.train = train
        self.full_image = full_image
        self.resize = resize

        # Pre-load all patches (or full images) into memory.
        self.samples = []
        self._prepare_dataset()

    def _prepare_dataset(self):
        for noisy_name, orig_name in zip(self.noisy_images, self.original_images):
            noisy_path = os.path.join(self.noisy_dir, noisy_name)
            orig_path = os.path.join(self.original_dir, orig_name)

            noisy_img = Image.open(noisy_path).convert("RGB")
            orig_img = Image.open(orig_path).convert("RGB")

            # Resize to a fixed size (256 x 256) before patch extraction
            noisy_img = noisy_img.resize(self.resize, Image.BICUBIC)
            orig_img = orig_img.resize(self.resize, Image.BICUBIC)

            # Convert to Tensor for easier patch extraction
            noisy_tensor = transforms.ToTensor()(noisy_img)
            orig_tensor = transforms.ToTensor()(orig_img)

            if self.full_image:
                # For testing, store the entire resized image
                self.samples.append((noisy_tensor, orig_tensor))
            else:
                # Extract patches for training/validation
                patches = self._extract_patches(noisy_tensor, orig_tensor,
                                                patch_size=self.patch_size,
                                                stride=self.stride)
                self.samples.extend(patches)

    def _extract_patches(self, noisy_tensor, orig_tensor, patch_size, stride):
        """
        Extract all patches of size patch_size with the given stride
        from noisy_tensor and orig_tensor.
        """
        _, H, W = noisy_tensor.shape
        patches = []
        for top in range(0, H - patch_size + 1, stride):
            for left in range(0, W - patch_size + 1, stride):
                noisy_patch = noisy_tensor[:, top:top+patch_size, left:left+patch_size]
                orig_patch = orig_tensor[:, top:top+patch_size, left:left+patch_size]
                patches.append((noisy_patch, orig_patch))
        return patches

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        noisy_img, orig_img = self.samples[idx]

        
        if self.train and self.transform is not None:
            
            noisy_img_pil = transforms.ToPILImage()(noisy_img)
            orig_img_pil = transforms.ToPILImage()(orig_img)

            noisy_img = self.transform(noisy_img_pil)
            orig_img = self.transform(orig_img_pil)

        return noisy_img, orig_img


train_augment = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])
no_augment = transforms.Compose([
    transforms.ToTensor()
])

################################################################################
#                              CREATE DATASETS
################################################################################

dataset_path = './dataset'

# Create train/val sets (patch-based) and test set (full image)
full_dataset = CBSD68PatchDataset(
    root_dir=dataset_path,
    patch_size=128,
    stride=64,
    train=True,
    transform=None,
    full_image=False,
    resize=(256, 256)
)

# Split dataset: 80% train, 10% val, 10% test
total_len = len(full_dataset)
train_size = int(0.8 * total_len)
val_size = int(0.1 * total_len)
test_size = total_len - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# Override transforms and flags for train/val sets
train_dataset.dataset.transform = train_augment
train_dataset.dataset.train = True

val_dataset.dataset.transform = no_augment
val_dataset.dataset.train = False

# For testing, create a separate dataset that returns full images
test_dataset_full = CBSD68PatchDataset(
    root_dir=dataset_path,
    patch_size=128,
    stride=64,
    train=False,
    transform=no_augment,
    full_image=True,
    resize=(256, 256)
)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset_full, batch_size=1, shuffle=False, num_workers=2)

################################################################################
#                              MODEL: U-NET
################################################################################

class DoubleConv(nn.Module):
    """
    Two consecutive convolution layers with BatchNorm + ReLU.
    """
    def __init__(self, in_channels, out_channels, dropout=False):
        super(DoubleConv, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(0.1))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    """
    U-Net with a single encoder/decoder level (features=[256]). The network output 
    is interpreted as the predicted noise for residual learning.
    """
    def __init__(self, in_channels=3, out_channels=3, features=[256]):
        super(UNet, self).__init__()
        self.encoder_layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder path
        prev_channels = in_channels
        for feature in features:
            self.encoder_layers.append(DoubleConv(prev_channels, feature))
            prev_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, dropout=True)

        # Decoder path
        self.up_transpose_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()

        decoder_channels = features[-1] * 2
        for feature in reversed(features):
            self.up_transpose_layers.append(
                nn.ConvTranspose2d(decoder_channels, feature, kernel_size=2, stride=2)
            )
            self.decoder_layers.append(
                DoubleConv(feature * 2, feature)
            )
            decoder_channels = feature

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        """
        For denoising (residual learning), the network output is the predicted noise.
        The denoised output is computed externally as: denoised = noisy - noise_pred.
        """
        skip_connections = []

        # Encoder
        for enc in self.encoder_layers:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder
        for idx in range(len(self.up_transpose_layers)):
            x = self.up_transpose_layers[idx](x)
            skip = skip_connections[idx]
            if x.shape != skip.shape:
                skip = self.center_crop(skip, x.shape[2], x.shape[3])
            x = torch.cat((skip, x), dim=1)
            x = self.decoder_layers[idx](x)

        # Final: predicted noise
        return self.final_conv(x)

    def center_crop(self, layer, target_height, target_width):
        _, _, h, w = layer.size()
        delta_h = (h - target_height) // 2
        delta_w = (w - target_width) // 2
        return layer[:, :, delta_h:delta_h+target_height, delta_w:delta_w+target_width]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)

################################################################################
#                              COMBINED LOSS
################################################################################

def combined_loss(noisy, noise_pred, clean):
    """
    Combined loss = MSE(denoised, clean) + (1 - SSIM(denoised, clean))

    Note: SSIM is computed via skimage and is non-differentiable. For a fully 
    differentiable solution, consider using pytorch-msssim.
    """
    # Compute denoised image using residual learning
    denoised = noisy - noise_pred
    denoised = torch.clamp(denoised, 0.0, 1.0)

    # MSE Loss
    mse_val = F.mse_loss(denoised, clean)

    # Compute SSIM in a non-differentiable manner (per-batch)
    denoised_np = denoised.detach().cpu().numpy()
    clean_np = clean.detach().cpu().numpy()

    ssim_batch = 0.0
    for b in range(denoised_np.shape[0]):
        ssim_batch += ssim(
            clean_np[b].transpose(1, 2, 0),
            denoised_np[b].transpose(1, 2, 0),
            win_size=7,
            data_range=1.0,
            channel_axis=-1
        )
    ssim_batch /= denoised_np.shape[0]
    ssim_loss = 1.0 - ssim_batch
    return mse_val + ssim_loss

################################################################################
#                              PSNR / SSIM
################################################################################

def calculate_psnr(original, denoised):
    mse = np.mean((original - denoised) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 1.0  # because images are in [0,1]
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def calculate_ssim_np(original, denoised):
    """
    Calculate SSIM with explicit win_size and channel_axis parameters.
    Both images are assumed to be in [0,1] with shape (H, W, C).
    """
    return ssim(
        original,
        denoised,
        win_size=7,
        data_range=1.0,
        channel_axis=-1
    )

################################################################################
#                            TRAINING / EVALUATION
################################################################################

def evaluate_metrics(model, loader, device):
    """
    Evaluate the model on a given loader, returning average combined loss,
    plus average PSNR and SSIM across the dataset.
    """
    model.eval()
    val_loss = 0.0
    psnr_total = 0.0
    ssim_total = 0.0
    count = 0

    with torch.no_grad():
        for noisy_imgs, clean_imgs in loader:
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)

            noise_pred = model(noisy_imgs)
            denoised = torch.clamp(noisy_imgs - noise_pred, 0.0, 1.0)

            loss_val = combined_loss(noisy_imgs, noise_pred, clean_imgs)
            val_loss += loss_val.item() * noisy_imgs.size(0)

            # Compute metrics per image in batch
            denoised_np = denoised.cpu().numpy()
            clean_np = clean_imgs.cpu().numpy()
            batch_size = denoised_np.shape[0]

            for b in range(batch_size):
                out_img = denoised_np[b].transpose(1, 2, 0)
                gt_img  = clean_np[b].transpose(1, 2, 0)
                psnr_total += calculate_psnr(gt_img, out_img)
                ssim_total += calculate_ssim_np(gt_img, out_img)
            count += batch_size

    avg_loss = val_loss / count
    avg_psnr = psnr_total / count
    avg_ssim = ssim_total / count
    return avg_loss, avg_psnr, avg_ssim

def train_model(model, train_loader, val_loader, device, num_epochs=50, lr=1e-4):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    train_losses = []
    val_losses = []
    val_psnrs = []
    val_ssims = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for noisy_imgs, clean_imgs in pbar:
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)

            optimizer.zero_grad()
            noise_pred = model(noisy_imgs)
            loss = combined_loss(noisy_imgs, noise_pred, clean_imgs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * noisy_imgs.size(0)
            num_samples += noisy_imgs.size(0)
            pbar.set_postfix(loss=loss.item())

        epoch_train_loss = running_loss / num_samples
        train_losses.append(epoch_train_loss)

        # Validation
        val_loss, val_psnr, val_ssim = evaluate_metrics(model, val_loader, device)
        val_losses.append(val_loss)
        val_psnrs.append(val_psnr)
        val_ssims.append(val_ssim)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val PSNR: {val_psnr:.2f} | "
              f"Val SSIM: {val_ssim:.4f}")

        # ------------------ W&B Logging ------------------
        wandb.log({
            "epoch": epoch+1,
            "train_loss": epoch_train_loss,
            "val_loss": val_loss,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim
        })
        # ---------------------------------------------------

        scheduler.step(val_loss)

    # Plot loss curves
    plt.figure(figsize=(8,6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title("Loss vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Plot PSNR and SSIM curves
    plt.figure(figsize=(8,6))
    plt.plot(val_psnrs, label='Val PSNR')
    plt.plot(val_ssims, label='Val SSIM')
    plt.title("PSNR and SSIM vs. Epoch")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()

    return train_losses, val_losses, val_psnrs, val_ssims

################################################################################
#                                RUN TRAINING
################################################################################

# Initialize Weights & Biases (W&B)
wandb.init(
    project="UNet-Image-Denoising",
        name="UNet_CBSD68",
    config={
        "num_epochs": 50,
        "lr": 1e-4,
        "batch_size": 4,
        "patch_size": 128,
        "stride": 64,
        "resize": (256, 256)
    }
)

# Log the model architecture
wandb.watch(model, log="all", log_freq=10)

num_epochs = 50  # Set desired number of epochs
train_model(model, train_loader, val_loader, device, num_epochs=num_epochs, lr=1e-4)

# Save model weights locally and log to W&B
torch.save(model.state_dict(), "best_unet_model.pth")
wandb.save("best_unet_model.pth")

################################################################################
#                        EVALUATE & VISUALIZE ON TEST SET
################################################################################

def visualize_results(noisy, denoised, original, psnr_val, ssim_val):
    """Visualizes noisy, denoised, and original images with PSNR and SSIM."""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(noisy)
    plt.title('Noisy Input')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(denoised)
    plt.title(f'Denoised\nPSNR: {psnr_val:.2f}, SSIM: {ssim_val:.2f}')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(original)
    plt.title('Ground Truth')
    plt.axis('off')

    plt.show()

def evaluate_and_visualize(model, dataloader, device, num_visualizations=5):
    model.eval()
    psnr_total, ssim_total, count = 0.0, 0.0, 0

    with torch.no_grad():
        for i, (noisy_imgs, clean_imgs) in enumerate(dataloader):
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)

            noise_pred = model(noisy_imgs)
            denoised = torch.clamp(noisy_imgs - noise_pred, 0, 1)

            noisy_np = noisy_imgs.cpu().numpy()
            denoised_np = denoised.cpu().numpy()
            clean_np = clean_imgs.cpu().numpy()

            batch_size = noisy_np.shape[0]
            for b in range(batch_size):
                n_img = noisy_np[b].transpose(1, 2, 0)
                d_img = denoised_np[b].transpose(1, 2, 0)
                c_img = clean_np[b].transpose(1, 2, 0)

                psnr_val = calculate_psnr(c_img, d_img)
                ssim_val = calculate_ssim_np(c_img, d_img)

                psnr_total += psnr_val
                ssim_total += ssim_val
                count += 1

                if i < num_visualizations and b == 0:
                    visualize_results(n_img, d_img, c_img, psnr_val, ssim_val)

    avg_psnr = psnr_total / count
    avg_ssim = ssim_total / count
    print(f"Test Average PSNR: {avg_psnr:.2f} dB")
    print(f"Test Average SSIM: {avg_ssim:.4f}")

# Finally, evaluate on the test set (full images)
evaluate_and_visualize(model, test_loader, device, num_visualizations=5)

wandb.finish()

