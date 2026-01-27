# BASIC PYTHON & ML IMPORTS

import os                          # Used to handle file paths and directory operations
import cv2                         # OpenCV library for image reading and processing
import numpy as np                 # NumPy for numerical computations and arrays
import pickle                      # Pickle for saving/loading Python objects to disk
import torch                       # Main PyTorch library for deep learning
import timm                        # timm provides pretrained models like EfficientNet
import torch.nn as nn              # PyTorch module for neural network layers
import random                      # Random module for reproducibility

# SEED SETUP
SEED = 42                          # Fixed seed value for reproducibility
random.seed(SEED)                  # Ensures Python random behaves the same every run
np.random.seed(SEED)               # Ensures NumPy random behaves the same every run
torch.manual_seed(SEED)            # Ensures PyTorch random behaves the same every run

import torch.optim as optim         # Optimizers like Adam
import matplotlib.pyplot as plt     # Used for visualization of images and masks
from torch.utils.data import Dataset, DataLoader  # Dataset & DataLoader utilities
from sklearn.metrics import f1_score, jaccard_score, accuracy_score  # Metrics for segmentation evaluation

# HYPERPARAMETERS & SETTINGS

IMG_SIZE = 256                     # Size to which all images are resized
BATCH_SIZE = 4                 # Number of images per batch
EPOCHS = 20                  # Number of training epochs
LR = 0.001                       # Learning rate for optimizer
THRESHOLD = 0.5                    # Threshold to convert probabilities to binary mask
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"    # Use GPU if available else CPU
DATASET_SIZE = 500             # Total number of samples used
DISEASE_RATIO = 0.6                # Percentage of disease samples
VIS_SAMPLES = 10       # Number of samples to visualize
SPLIT_PICKLE_PATH = r"D:\BT\split.pkl" # Path to save train-test split indices
BEST_MODEL_PATH = "Best_model.pth" # Path to save best trained model

# Dataset
class BrainMRIDataset(Dataset):     # Custom Dataset for MRI images
    def __init__(self, root_dir, img_size=IMG_SIZE, total_samples=DATASET_SIZE, disease_ratio=DISEASE_RATIO, seed=SEED , debug=True, debug_samples= 4 ):
        self.img_size = img_size    # Store image size
        self.samples = []           # List to store (image, mask, label)
        self.labels = []            # Store labels separately for statistics

        disease_dir = os.path.join(root_dir, "disease")        # Folder containing disease images
        normal_dir = os.path.join(root_dir, "normal")                # Folder containing normal images

        disease_samples = self._load_pairs(disease_dir, label=1)        # Load disease images with label 1
        normal_samples = self._load_pairs(normal_dir, label=0)        # Load normal images with label 0

        rng = np.random.default_rng(seed)        # NumPy random generator for reproducibility

        rng.shuffle(disease_samples)  # Shuffle disease samples
        rng.shuffle(normal_samples)   # Shuffle normal samples

        d_count = int(total_samples * disease_ratio)      # Number of disease images
        n_count = total_samples - d_count                                    # Number of normal images

        disease_samples = disease_samples[:d_count]                # Select required disease samples
        normal_samples = normal_samples[:n_count]            # Select required normal samples

        combined = disease_samples + normal_samples          # Combine disease and normal samples
        rng.shuffle(combined)        # Shuffle final dataset

        for img, mask, label in combined:
            self.samples.append((img, mask, label))       # Store image-mask-label tuple
            self.labels.append(label)               # Store label separately

        print(f"Total samples  : {len(self.samples)}")
        print(f"Disease images : {sum(self.labels)}")
        print(f"Normal images  : {len(self.labels) - sum(self.labels)}")

        if len(self.samples) == 0:   # Safety check
            raise ValueError("No image-mask pairs found.")

    def _load_pairs(self, folder, label):
        pairs = []                  # List to store image-mask pairs
        for f in os.listdir(folder):                                    # Loop through folder files
            if f.endswith(".tif") and not f.endswith("_mask.tif"):                                    # Ensure image file (not mask)
                img_path = os.path.join(folder, f)                                    # Full image path
                mask_path = os.path.join(folder, f.replace(".tif", "_mask.tif"))       # Corresponding mask path
                if os.path.exists(mask_path):
                                    # Ensure mask exists
                    pairs.append((img_path, mask_path, label)) # Add valid pair
        return pairs

    def __len__(self):
        return len(self.samples)     # Return dataset size

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]      # Get sample paths and label

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # Read MRI image as grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  
        # self._show(image, "Raw Image")
        # self._show(mask, "Raw Mask")  # Read mask as grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  
        # self._show(image[..., 0], "Grcl-rgb")    # Convert to 3-channel image
        image = cv2.resize(image, (self.img_size, self.img_size))  # Resize image
        mask = cv2.resize(mask, (self.img_size, self.img_size),
            interpolation=cv2.INTER_NEAREST)  
        # self._show(image[..., 0], "Resized Image")
        # self._show(mask, "Resized Mask") # Resize mask without smoothing

        image = image / 255.0        # Normalize image
        # self._show(image[..., 0], "Normalized Image")
        mask = (mask > 0).astype(np.float32)  # Convert mask to binary
        # self._show(mask, "Binary Mask")

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)    # Convert to tensor and rearrange channels
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)     # Add channel dimension
        label = torch.tensor(label, dtype=torch.long) # Convert label to tensor

        return image, mask, label 
    
    # def _show(self, img, title):
    #     plt.figure(figsize=(4, 4))
    #     plt.imshow(img, cmap="gray")
    #     plt.title(title)
    #     plt.axis("off")
    #     plt.show() 
    
# Dataset from Pickle (for exact reproducibility)

# Custom Dataset class that loads data from a pickle file
# Used to ensure the same train/test data is reused every time
class PickleDataset(Dataset):

    def __init__(self, pickle_file, split="train"):
        # Constructor method called when the dataset object is created
        # pickle_file - path to the saved dataset
        # split - specifies whether to load 'train' or 'test' data
        
        with open(pickle_file, "rb") as f:             # Open the pickle file in binary read mode
            data = pickle.load(f)            # Load the dictionary containing saved tensors
        self.pkl = pickle_file
        self.images = data[f"{split}_images"]        # Load image tensors for the selected split (train/test)
        self.masks = data[f"{split}_masks"]        # Load corresponding segmentation masks

        self.labels = data[f"{split}_labels"]       # Load class labels (0 = normal, 1 = disease)

    def img_len(self):
        # This method returns the total number of samples in the dataset
        # Required by PyTorch DataLoader
        return len(self.images)         # Dataset length is based on number of images
    
    def img_list(self):
        # This method returns the total number of samples in the dataset
        # Required by PyTorch DataLoader
        return self.images        
    
    def __len__(self):
        # Required by DataLoader, simply call img_len()
        return self.img_len()
    
    def __getitem__(self, idx):
        # This method returns one sample given its index
        # DataLoader uses this to fetch data during training

        return self.images[idx], self.masks[idx], self.labels[idx]  # Return image, mask, and label as a tuple
    
def visualize_pickle_images(pickle_file, split="train", num_samples=3):

    with open(pickle_file, "rb") as f:
        data = pickle.load(f)

    images = data[f"{split}_images"] 
    masks = data[f"{split}_masks"]    
    labels = data[f"{split}_labels"] 

    for i in range(min(num_samples, len(images))):
        img = images[i]
        mask = masks[i]
        label = labels[i].item() 

        if torch.is_tensor(img):
            img = img.cpu().permute(1, 2, 0).numpy()  
        if torch.is_tensor(mask):
            mask = mask.cpu().squeeze(0).numpy()     

        plt.figure(figsize=(8, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(img[..., 0], cmap="gray")
        plt.title("Train images")
        plt.title(f"Image - Label: {label}")
        plt.axis("off")

        # Show mask
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap="gray")
        plt.title("Mask")
        plt.axis("off")

        plt.show()

visualize_pickle_images("split_data.pkl", split="train", num_samples=4)

# UNet Architecture
# BASIC CONVOLUTION BLOCK
class ConvBlock(nn.Module):   # Defines a reusable convolutional block used in the decoder

    def __init__(self, in_c, out_c):
        # in_c - number of input channels
        # out_c - number of output channels
        super().__init__()
        # Initialize the parent nn.Module class

        self.conv = nn.Sequential(
            # Sequential container to apply layers in order

            nn.Conv2d(in_c, out_c, 3, padding=1),
            # First convolution layer with 3×3 kernel
            # padding=1 keeps spatial size same

            nn.BatchNorm2d(out_c),           # Batch normalization for stable and faster training

            nn.ReLU(inplace=True),         # ReLU activation introduces non-linearity

            nn.Conv2d(out_c, out_c, 3, padding=1),       # Second convolution layer for better feature learning

            nn.BatchNorm2d(out_c),        # Normalize features again

            nn.ReLU(inplace=True),        # Activation after second convolution
        )

    def forward(self, x):      # Forward pass of ConvBlock
        return self.conv(x)    # Apply all layers sequentially to input x

# SKIP CONNECTION CONV BLOCK

class SkipConv(nn.Module):   # Lightweight convolution block for skip connections

    def __init__(self, in_c, out_c):
        # in_c - input feature channels from encoder
        # out_c - output channels for decoder compatibility
        super().__init__()
        # Initialize parent class

        self.block = nn.Sequential(
            # Sequential container

            nn.Conv2d(in_c, out_c, 3, padding=1),       # Convolution to adjust channel size

            nn.BatchNorm2d(out_c),           # Normalize encoder features

            nn.ReLU(inplace=True),            # Activation function
        )
    def forward(self, x):
        # Forward pass for skip connection
        return self.block(x)
        # Process and return skip features

# EFFICIENTNET ENCODER
class EfficientNetEncoder(nn.Module):   # Encoder based on pretrained EfficientNet-B7

    def __init__(self):
        super().__init__()    # Initialize parent class

        self.model = timm.create_model(
            "tf_efficientnet_b7_ns",
            pretrained=True,
            features_only=True
        )
        # Load pretrained EfficientNet-B7
        # features_only=True returns intermediate feature maps

    def forward(self, x):     # Forward pass of encoder
        return self.model(x)    # Output multi-scale feature maps

# EFFICIENT U-NET MODEL

class EfficientUNet(nn.Module):    # U-Net architecture using EfficientNet encoder

    def __init__(self, out_ch=1):  # out_ch - number of output channels (1 for binary segmentation)
        super().__init__()        # Initialize parent class

        self.encoder = EfficientNetEncoder()       # Backbone encoder for feature extraction

        self.skip0 = SkipConv(32, 64)    # Skip connection for first encoder level

        self.skip1 = SkipConv(48, 128)
        # Skip connection for second encoder level

        self.skip2 = SkipConv(80, 256)
        # Skip connection for third encoder level

        self.skip3 = SkipConv(224, 512)
        # Skip connection for fourth encoder level

        self.skip4 = SkipConv(640, 640)
        # Skip connection for deepest encoder features

        self.up = nn.Upsample( scale_factor=2,mode="bilinear",  align_corners=True)
        # Upsampling layer used in decoder

        self.dec3 = ConvBlock(640 + 512, 512)        # Decoder block combining encoder + upsampled features

        self.dec2 = ConvBlock(512 + 256, 256)      # Decoder block for next level

        self.dec1 = ConvBlock(256 + 128, 128)     # Decoder block for next level

        self.dec0 = ConvBlock(128 + 64, 64)     # Final decoder block

        self.final = nn.Conv2d(64, out_ch, 1)  # Final 1×1 convolution to produce segmentation mask

    def forward(self, x):      # Forward pass of the full U-Net

        feats = self.encoder(x)       # Extract multi-scale features from encoder

        e0 = self.skip0(feats[0])      # Process first encoder feature map

        e1 = self.skip1(feats[1])    # Process second encoder feature map

        e2 = self.skip2(feats[2])    # Process third encoder feature map

        e3 = self.skip3(feats[3])    # Process fourth encoder feature map

        e4 = self.skip4(feats[4])     # Process deepest encoder feature map

        d3 = self.dec3(torch.cat([e3, self.up(e4)], dim=1)) # Upsample e4, concatenate with e3, then decode

        d2 = self.dec2(torch.cat([e2, self.up(d3)], dim=1)) # Upsample d3, concatenate with e2, then decode

        d1 = self.dec1(torch.cat([e1, self.up(d2)], dim=1))   #  d2, relate with e1, then decode

        d0 = self.dec0(torch.cat([e0, self.up(d1)], dim=1))  # Upsample d1, concatenate with e0, then decode

        out = self.final(d0)     # Generate raw segmentation logits

        return nn.functional.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)     # Resize output to match input image size

# Losses and Metrics


bce_loss = nn.BCEWithLogitsLoss()# Binary Cross Entropy loss with logits
# Used for pixel-wise binary classification
# Combines sigmoid activation + BCE loss internally for numerical stability


def dice_loss(pred, target, eps=1e-6):    # Dice loss function used for segmentation overlap accuracy
    # pred   - raw model outputs (logits)
    # target -ground truth segmentation masks
    # eps    - small value to avoid division by zero

    pred = torch.sigmoid(pred).view(pred.size(0), -1)    # Apply sigmoid to convert logits into probabilities
    # Flatten predictions to (batch_size, num_pixels)

    target = target.view(target.size(0), -1)
    # Flatten ground truth masks to match prediction shape

    inter = (pred * target).sum(dim=1)
    # Compute intersection between prediction and ground truth

    union = pred.sum(dim=1) + target.sum(dim=1)
    # Compute union of prediction and ground truth

    dice = (2 * inter + eps) / (union + eps)
    # Compute Dice coefficient for each sample

    return 1 - dice.mean()
    # Dice loss = 1 − Dice score (averaged across batch)


def combined_loss(pred, target):
    # Combined loss function to balance pixel accuracy and region overlap

    return 0.5 * bce_loss(pred, target) + dice_loss(pred, target)
    # Weighted sum of BCE loss and Dice loss
    # BCE handles pixel-level correctness
    # Dice improves segmentation shape accuracy


def segmentation_metrics(pred, target):    # Function to compute evaluation metrics for segmentation

    pred_bin = (torch.sigmoid(pred) > THRESHOLD).float()
    # Convert logits to probabilities and then to binary mask using threshold

    pred_np = pred_bin.cpu().numpy().reshape(-1)    # Move prediction to CPU, convert to NumPy, and flatten

    target_np = target.cpu().numpy().reshape(-1)
    # Move ground truth to CPU, convert to NumPy, and flatten

    acc = accuracy_score(target_np, pred_np)
    # Compute pixel-wise accuracy

    dice = f1_score(target_np, pred_np, zero_division=0)    # Compute Dice score (F1 score for binary segmentation)

    iou = jaccard_score(target_np, pred_np, zero_division=0)
    # Compute Intersection over Union (IoU)

    return dice, iou, acc
    # Return all segmentation metrics

# Visualization
def visualize_first_samples(model, dataset, device, num_samples=5):
    # Function to visualize original MRI, ground truth mask, and predicted mask
    # model       - trained segmentation model
    # dataset     - dataset to visualize samples from
    # device      - CPU or GPU
    # num_samples - number of samples to display

    model.eval()    # Set model to evaluation mode (disables dropout, uses running batch norm stats)

    shown = 0    # Counter to track how many samples have been visualized

    idx = 0    # Dataset index pointer

    with torch.no_grad():        # Disable gradient computation for faster inference and lower memory usage

        while shown < num_samples and idx < len(dataset):            # Loop until required number of samples are shown
            # or dataset is exhausted

            image, mask, label = dataset[idx]            # Retrieve one image, mask, and label from dataset

            idx += 1            # Move to next dataset index

            image = image.unsqueeze(0).to(device)
            # Add batch dimension and move image to CPU/GPU

            mask = mask.unsqueeze(0).to(device)            # Add batch dimension and move mask to CPU/GPU

            pred = model(image)            # Forward pass through the model to get prediction logits

            pred_bin = (torch.sigmoid(pred) > THRESHOLD).float()            # Convert logits to probabilities and then to binary mask

            img_np = image[0].cpu().permute(1, 2, 0).numpy()            # Convert image tensor to NumPy array for visualization
            # Change shape from (C, H, W) to (H, W, C)

            gt_np = mask[0, 0].cpu().numpy()            # Extract ground truth mask and convert to NumPy

            pred_np = pred_bin[0, 0].cpu().numpy()            # Extract predicted mask and convert to NumPy

            plt.figure(figsize=(12, 4))            # Create a new figure for visualization

            plt.subplot(1, 3, 1)            # First subplot for original MRI image

            plt.imshow(img_np, cmap="gray")            # Display original MRI image

            plt.title("Original MRI")            # Title for original image

            plt.axis("off")            # Hide axis ticks and labels

            plt.subplot(1, 3, 2)            # Second subplot for ground truth mask

            plt.imshow(gt_np, cmap="gray")            # Display ground truth segmentation mask

            plt.title("Ground Truth Mask")            # Title for ground truth mask

            plt.axis("off")            # Hide axes

            plt.subplot(1, 3, 3)            # Third subplot for predicted mask

            plt.imshow(pred_np, cmap="gray")            # Display predicted segmentation mask

            plt.title("Predicted Mask")            # Title for predicted mask

            plt.axis("off")            # Hide axes

            plt.tight_layout()            # Automatically adjust subplot spacing

            plt.show()            # Render the figure on screen

            shown += 1    


# def visualize_unet_input(img_batch, mask_batch=None, max_samples=2):
    
#     img_batch = img_batch.cpu()
#     if mask_batch is not None:
#         mask_batch = mask_batch.cpu()

#     for i in range(min(max_samples, img_batch.size(0))):
#         img = img_batch[i].permute(1, 2, 0).numpy()  # (H, W, C)
#         img_gray = img[..., 0]  # MRI is grayscale replicated to 3 channels

#         plt.figure(figsize=(8, 4))

#         plt.subplot(1, 2 if mask_batch is not None else 1, 1)
#         plt.imshow(img_gray, cmap="gray")
#         plt.title("UNet Input Image")
#         plt.axis("off")

#         if mask_batch is not None:
#             mask = mask_batch[i, 0].numpy()
#             plt.subplot(1, 2, 2)
#             plt.imshow(mask, cmap="gray")
#             plt.title("Ground Truth Mask")
#             plt.axis("off")

#         plt.tight_layout()
#         plt.show()



# Save split indices

def save_split_pickle(train_indices, test_indices, pickle_path):    # Function to save train and test indices for reproducibility

    split_dict = {"train_indices": train_indices, "test_indices": test_indices}    # Create a dictionary containing train and test indices

    with open(pickle_path, "wb") as f:        # Open a file in binary write mode
        pickle.dump(split_dict, f)        # Save the split dictionary into the pickle file

    print(f"Train/test split saved to {pickle_path}")    # Print confirmation message

# MODEL EVALUATION FUNCTION

def evaluate(model, dataloader, device):    # Function to evaluate model performance on validation/test data

    model.eval()    # Set model to evaluation mode (disables dropout, fixes batch norm)

    total_acc = total_dice = total_iou = 0.0    # Initialize accumulators for metrics

    steps = 0    # Counter to track number of batches

    with torch.no_grad():# Disable gradient computation for faster inference

        for img, mask, label in dataloader: # Iterate through batches from dataloader

            img, mask, label = img.to(device), mask.to(device), label.to(device)# Move data to CPU or GPU

            pred = model(img)            # Forward pass to get model predictions

            dice, iou, acc = segmentation_metrics(pred, mask)      # Compute segmentation metrics

            total_acc += acc           # Accumulate accuracy

            total_dice += dice   # Accumulate Dice score

            total_iou += iou  # Accumulate IoU score

            steps += 1 # Increment batch counter

    if steps == 0:  # Safety check in case dataloader is empty
        return 0, 0, 0

    return total_acc / steps, total_dice / steps, total_iou / steps  # Return average metrics across all batches

# TRAINING FUNCTION

def train():  # Main training function

    dataset = BrainMRIDataset( root_dir=r"D:\BT\dataset",  total_samples=DATASET_SIZE, disease_ratio=DISEASE_RATIO, seed=SEED)
    # Load and preprocess dataset from disk

    rng = np.random.default_rng(SEED)    # Create reproducible random generator

    indices = np.arange(len(dataset))    # Create array of dataset indices

    rng.shuffle(indices)    # Shuffle indices randomly
    

    split = int(0.8 * len(indices))  # Define 80% split point

    train_indices = indices[:split] # Indices for training set

    test_indices = indices[split:]   # Indices for test set

    def save_data(indices, split_name):    # Helper function to extract tensors and store them

        images, masks, labels = [], [], []  # Lists to store image, mask, and label tensors

        for idx in indices:     # Loop through given indices

            img, mask, label = dataset[idx] # Load sample from dataset

            images.append(img)  # Store image tensor

            masks.append(mask)           # Store mask tensor

            labels.append(label)     # Store label tensor

        return torch.stack(images), torch.stack(masks), torch.tensor(labels)   # Stack lists into tensors and return

    train_images, train_masks, train_labels = save_data(train_indices, "train")  # Save training data tensors

    test_images, test_masks, test_labels = save_data(test_indices, "test") # Save testing data tensors

    with open("split_data.pkl", "wb") as f:   # Open pickle file to store tensors

        pickle.dump({
            "train_images": train_images,
            "train_masks": train_masks,
            "train_labels": train_labels,
            "test_images": test_images,
            "test_masks": test_masks,
            "test_labels": test_labels
        }, f)    # Save all tensors into one pickle file

    print("Train and test images saved to split_data.pkl") # Print confirmation message

    save_split_pickle(train_indices, test_indices, SPLIT_PICKLE_PATH)  # Save train/test indices separately for reproducibility

    obj1 = PickleDataset("split_data.pkl", "train")

    train_loader = DataLoader( obj1, batch_size=BATCH_SIZE, shuffle=True, ) # DataLoader for training data with shuffling
    print(obj1.img_len)

    obj2 = PickleDataset("split_data.pkl", "test")

    test_loader = DataLoader(obj2,  batch_size=BATCH_SIZE,  shuffle=False )  # DataLoader for test data without shuffling

    model = EfficientUNet().to(DEVICE) # Initialize model and move to CPU/GPU

    optimizer = optim.Adam(model.parameters(), lr=LR) # Adam optimizer for training

    best_dice = 0.0 # Variable to track best Dice score

    for img, mask, label in train_loader:
 
            img, mask, label = img.to(DEVICE), mask.to(DEVICE), label.to(DEVICE)
            # visualize_unet_input(img)   # THIS is the UNet input
            pred = model(img)



    for epoch in range(EPOCHS): # Loop over epochs

        model.train()   # Set model to training mode

        total_loss = total_acc = total_dice = total_iou = 0.0  # Reset metric accumulators

        steps = 0 # Reset step counter

        for img, mask, label in train_loader:            # Loop over training batches

            img, mask, label = img.to(DEVICE), mask.to(DEVICE), label.to(DEVICE) # Move batch to CPU/GPU

            pred = model(img)  # Forward pass

            loss = combined_loss(pred, mask)  # Compute loss

            optimizer.zero_grad()  # Clear previous gradients

            loss.backward()# Backpropagate loss

            optimizer.step() # Update model parameters

            dice, iou, acc = segmentation_metrics(pred, mask)  # Compute training metrics

            total_loss += loss.item() # Accumulate loss

            total_acc += acc  # Accumulate accuracy

            total_dice += dice  # Accumulate Dice score

            total_iou += iou  # Accumulate IoU score

            steps += 1  # Increment batch counter

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"Loss: {total_loss/steps:.4f} | "
            f"Acc: {total_acc/steps:.4f} | "
            f"Dice: {total_dice/steps:.4f} | "
            f"IoU: {total_iou/steps:.4f}"
        )   # Print training statistics for the epoch

        if total_dice / steps > best_dice: # Check if current model is better

            best_dice = total_dice / steps  # Update best Dice score

            torch.save({  "epoch": epoch + 1,"model_state_dict": model.state_dict(), "best_dice": best_dice}, BEST_MODEL_PATH)    # Save model checkpoint
            print(f"New best model saved at epoch {epoch+1}")  # Print confirmation message

    test_acc, test_dice, test_iou = evaluate(model, test_loader, DEVICE) # Evaluate model on test set

    print(f"\nTesting | Acc: {test_acc:.4f} | Dice: {test_dice:.4f} | IoU: {test_iou:.4f}")  # Print test performance metrics

    visualize_first_samples( model, PickleDataset("split_data.pkl", "test"),DEVICE, VIS_SAMPLES) # Visualize predicted masks vs ground truth

if __name__ == "__main__":# Ensures training runs only when script is executed directly
    train()# Start training

