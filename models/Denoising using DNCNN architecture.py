#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# In[8]:


import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm  

# ------------------ NEW: Import Weights & Biases ------------------
import wandb
# ------------------------------------------------------------------

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

###############################################################################
#                          DENOISE DATASET (UPDATED)
###############################################################################
class DenoiseDataset(Dataset):
    """
    If full_image=False, extracts patches with a given patch_size and stride.
    If full_image=True, loads entire images without patch extraction.
    """
    def __init__(
        self,
        original_dir,
        patch_size=50,
        stride=10,
        train=True,
        noise_level=35,
        full_image=False
    ):
        self.original_dir = original_dir
        self.patch_size = patch_size
        self.stride = stride
        self.train = train
        self.noise_level = noise_level
        self.full_image = full_image
        
        def is_image_file(filename):
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
            return any(filename.lower().endswith(ext) for ext in valid_extensions)
        
        self.image_files = [
            f for f in sorted(os.listdir(original_dir))
            if os.path.isfile(os.path.join(original_dir, f))
            and is_image_file(f)
            and not f.startswith('.')
        ]
        
        # If we are doing patch extraction, we store patches in self.patches.
        # If we are loading full images, we won't do patch extraction.
        self.patches = []
        if not self.full_image:
            self._extract_patches()

    def _extract_patches(self):
        for img_file in self.image_files:
            img_path = os.path.join(self.original_dir, img_file)
            try:
                img = Image.open(img_path).convert('RGB')
                img = np.array(img) / 255.0  # Normalize to [0, 1]
                h, w, _ = img.shape
                for i in range(0, h - self.patch_size + 1, self.stride):
                    for j in range(0, w - self.patch_size + 1, self.stride):
                        patch = img[i:i+self.patch_size, j:j+self.patch_size, :]
                        self.patches.append(patch)
            except Exception as e:
                print(f"Error processing {img_file}: {e}")

    def __len__(self):
        # If full_image=True, length = number of image files.
        # Otherwise, length = number of patches.
        if self.full_image:
            return len(self.image_files)
        else:
            return len(self.patches)
    
    def __getitem__(self, idx):
        if self.full_image:
            # Load the entire image
            img_file = self.image_files[idx]
            img_path = os.path.join(self.original_dir, img_file)
            img = Image.open(img_path).convert('RGB')
            img = np.array(img) / 255.0  # Normalize to [0, 1]
            clean_patch = torch.from_numpy(img.transpose((2, 0, 1))).float()
        else:
            # Load a patch
            patch = self.patches[idx]
            clean_patch = torch.from_numpy(patch.transpose((2, 0, 1))).float()
        
        # Add noise
        noise = torch.randn_like(clean_patch) * (self.noise_level / 255.0)
        noisy_patch = torch.clamp(clean_patch + noise, 0, 1)
        
        # Data augmentation only for training patches
        if self.train and not self.full_image:
            if torch.rand(1) < 0.5:
                clean_patch = clean_patch.flip(2)
                noisy_patch = noisy_patch.flip(2)
            if torch.rand(1) < 0.5:
                clean_patch = clean_patch.flip(1)
                noisy_patch = noisy_patch.flip(1)
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                clean_patch = torch.rot90(clean_patch, k, [1, 2])
                noisy_patch = torch.rot90(noisy_patch, k, [1, 2])
        
        return noisy_patch, clean_patch

###############################################################################
#                       DnCNN Model + Losses + Metrics
###############################################################################
class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=3, use_bnorm=True):
        super(DnCNN, self).__init__()
        
        # First layer: Conv + ReLU
        layers = [
            nn.Conv2d(image_channels, n_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        ]
        
        # Middle layers: Conv + BN + ReLU
        for _ in range(depth - 2):
            layers.extend([
                nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(n_channels) if use_bnorm else nn.Identity(),
                nn.ReLU(inplace=True)
            ])
        
        # Last layer: Conv
        layers.append(nn.Conv2d(n_channels, image_channels, kernel_size=3, padding=1, bias=False))
        
        self.dncnn = nn.Sequential(*layers)
    
    def forward(self, x):
        noise = self.dncnn(x)
        return x - noise  # Residual learning

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel = 3
        self.window = self._create_window(window_size)
        
    def _create_window(self, window_size):
        sigma = 1.5
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/(2*sigma**2)) for x in range(window_size)])
        gauss = gauss / gauss.sum()
        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(self.channel, 1, window_size, window_size).contiguous()
        return window
        
    def forward(self, img1, img2):
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)
            
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=self.channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size//2, groups=self.channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.8):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIMLoss()
    
    def forward(self, x, y):
        return self.alpha * self.mse_loss(x, y) + (1 - self.alpha) * self.ssim_loss(x, y)

def calculate_metrics(clean_img, denoised_img):
    if isinstance(clean_img, torch.Tensor):
        clean_img = clean_img.detach().cpu().numpy()
    if isinstance(denoised_img, torch.Tensor):
        denoised_img = denoised_img.detach().cpu().numpy()
    if clean_img.ndim == 3 and clean_img.shape[0] == 3:
        clean_img = np.transpose(clean_img, (1, 2, 0))
        denoised_img = np.transpose(denoised_img, (1, 2, 0))
    psnr_value = psnr(clean_img, denoised_img, data_range=1.0)
    try:
        win_size = min(7, min(clean_img.shape[0], clean_img.shape[1])-1)
        win_size = win_size if win_size % 2 == 1 else win_size - 1
        ssim_value = ssim(clean_img, denoised_img, data_range=1.0, channel_axis=-1, win_size=win_size)
    except Exception as e:
        print(f"SSIM calculation error: {e}")
        ssim_value = ssim(clean_img, denoised_img, data_range=1.0, channel_axis=-1, win_size=3)
    return psnr_value, ssim_value

def visualize_results(noisy, denoised, original, psnr_val, ssim_val):
    """Visualizes noisy, denoised, and original images side-by-side."""
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

###############################################################################
#                    TRAINING LOOP (with early stopping + W&B)
###############################################################################
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                patience=10, max_epochs=100, master_bar=None):
    best_val_psnr = 0
    best_val_ssim = 0
    best_model_state = None
    best_epoch = 0
    no_improve_epochs = 0
    
    training_history = {
        'train_loss': [],
        'val_psnr': [],
        'val_ssim': []
    }
    
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for noisy_imgs, clean_imgs in train_loader:
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
            optimizer.zero_grad()
            denoised_imgs = model(noisy_imgs)
            loss = criterion(denoised_imgs, clean_imgs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        training_history['train_loss'].append(avg_train_loss)
        
        model.eval()
        val_psnr_list = []
        val_ssim_list = []
        with torch.no_grad():
            for noisy_imgs, clean_imgs in val_loader:
                noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
                denoised_imgs = model(noisy_imgs)
                for i in range(noisy_imgs.size(0)):
                    psnr_val, ssim_val = calculate_metrics(clean_imgs[i], denoised_imgs[i])
                    val_psnr_list.append(psnr_val)
                    val_ssim_list.append(ssim_val)
        avg_val_psnr = np.mean(val_psnr_list)
        avg_val_ssim = np.mean(val_ssim_list)
        training_history['val_psnr'].append(avg_val_psnr)
        training_history['val_ssim'].append(avg_val_ssim)
        
        print(f"Epoch {epoch+1}: Loss = {avg_train_loss:.6f}, PSNR = {avg_val_psnr:.2f}, SSIM = {avg_val_ssim:.4f}")
        
        # Log metrics to W&B
        wandb.log({
            "epoch": epoch+1,
            "train_loss": avg_train_loss,
            "val_psnr": avg_val_psnr,
            "val_ssim": avg_val_ssim
        })
        
        scheduler.step(avg_val_psnr)
        if avg_val_psnr > best_val_psnr:
            best_val_psnr = avg_val_psnr
            best_val_ssim = avg_val_ssim
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
            no_improve_epochs = 0
            torch.save(model.state_dict(), 'best_dncnn_model.pth')
            print(f"New best model saved with PSNR: {best_val_psnr:.2f} dB")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if master_bar is not None:
            master_bar.update(1)
    
    model.load_state_dict(best_model_state)
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(training_history['train_loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.subplot(1, 3, 2)
    plt.plot(training_history['val_psnr'])
    plt.title('Validation PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.subplot(1, 3, 3)
    plt.plot(training_history['val_ssim'])
    plt.title('Validation SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    return model, best_val_psnr, best_val_ssim, best_epoch

###############################################################################
#                              MAIN FUNCTION
###############################################################################
def main():
    torch.manual_seed(0)
    np.random.seed(0)
    
    original_dir = "./dataset/original_png" 
    
    # ------------------ TRAIN & VAL: patch-based  ------------------
    train_dataset = DenoiseDataset(
        original_dir=original_dir,
        patch_size=50,
        stride=20,
        train=True,
        noise_level=35,
        full_image=False   # <--- Patches for training
    )
    val_dataset = DenoiseDataset(
        original_dir=original_dir,
        patch_size=50,
        stride=40,
        train=False,
        noise_level=35,
        full_image=False   # <--- Patches for validation
    )
    
    # ------------------ TEST: full images  ------------------
    test_dataset = DenoiseDataset(
        original_dir=original_dir,
        train=False,
        noise_level=35,
        full_image=True    # <--- Full images for testing
    )
    
    # ------------------ Initialize W&B ------------------
# ------------------ Initialize W&B ------------------
    wandb.init(
        project="DnCNN-Image-Denoising",
        name="DnCNN_CBSD68_Batch16_Depth17",
        config={
            "max_epochs": 100,      
            "batch_size": 16,       
            "depth": 17,            
            "n_channels": 64,       
            "lr": 0.0005,           
            "weight_decay": 0.0001, 
            "noise_level": 35,      
            "patch_size": 50,       
            "stride": 20            
        }
    )
    # -----------------------------------------------------

    # Create DataLoaders
    batch_size = 16  
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True
    )

    
    
    from math import ceil
    max_epochs = 100
    training_steps = max_epochs
    test_steps = len(test_loader)
    overall_total = training_steps + test_steps
    master_bar = tqdm(total=overall_total, desc="Overall Progress", unit="step")
    
    # Create model
    model = DnCNN(depth=17, n_channels=64).to(device)
    
    # Watch the model with W&B
    wandb.watch(model, log="all", log_freq=10)
    
    # Define loss
    criterion = CombinedLoss(alpha=0.8)
    
    # Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    print("\nTraining model with patches (DnCNN)...")
    model, final_psnr, final_ssim, final_epoch = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        patience=10, max_epochs=100, master_bar=master_bar
    )
    
    print("\nEvaluating on test set (full images)...")
    model.eval()
    test_psnr_list = []
    test_ssim_list = []
    test_display_count = 0
    with torch.no_grad():
        for i, (noisy_imgs, clean_imgs) in enumerate(test_loader):
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
            denoised_imgs = model(noisy_imgs)
            psnr_val, ssim_val = calculate_metrics(clean_imgs[0], denoised_imgs[0])
            test_psnr_list.append(psnr_val)
            test_ssim_list.append(ssim_val)
            
            # Visualize the first few test images
            if test_display_count < 5:
                n_img = noisy_imgs[0].cpu().numpy().transpose(1, 2, 0)
                d_img = denoised_imgs[0].cpu().numpy().transpose(1, 2, 0)
                c_img = clean_imgs[0].cpu().numpy().transpose(1, 2, 0)
                visualize_results(n_img, d_img, c_img, psnr_val, ssim_val)
                test_display_count += 1
            
            master_bar.update(1)
    
    avg_test_psnr = np.mean(test_psnr_list)
    avg_test_ssim = np.mean(test_ssim_list)
    
    print(f"\nTest results - PSNR: {avg_test_psnr:.2f} dB, SSIM: {avg_test_ssim:.4f}")
    
    # Log final test metrics to W&B
    wandb.log({
        "test_psnr": avg_test_psnr,
        "test_ssim": avg_test_ssim
    })
    
    master_bar.close()
    print("Done!")
    wandb.finish()

if __name__ == "__main__":
    main()

