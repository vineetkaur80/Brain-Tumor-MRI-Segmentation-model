import pickle # Used to load saved data (images, masks) from a pickle file
import torch # Core PyTorch library for tensors and deep learning
import torch.nn as nn # Provides neural network modules and loss functions
import matplotlib.pyplot as plt # Used for visualizing images and masks
from torch.utils.data import TensorDataset, DataLoader # TensorDataset wraps tensors into a dataset
# DataLoader helps load data in batches
from sklearn.metrics import f1_score, jaccard_score, accuracy_score # Metrics for segmentation evaluation (Dice, IoU, Accuracy)
from trial import EfficientUNet, THRESHOLD, DEVICE # Import model architecture, threshold for binarization, and device (CPU/GPU)

# CONSTANTS / PARAMETERS
BATCH_SIZE = 2 # Number of samples processed at once
NUM_VISUALS = 5 # Number of samples to visualize
PICKLE_PATH = "split_data.pkl" # Path to pickle file containing test images and masks
MODEL_PATH = "200best_model.pth" # Path to saved trained model checkpoint

# METRICS FUNCTION

def segmentation_metrics(pred, target):     # Function to compute Dice, IoU, and Accuracy for segmentation
    pred_bin = (torch.sigmoid(pred) > THRESHOLD).float()     # Apply sigmoid to logits, threshold to binary mask
    pred_np = pred_bin.cpu().numpy().reshape(-1)     # Move predictions to CPU, flatten to 1D numpy array
    target_np = target.cpu().numpy().reshape(-1)     # Move ground truth to CPU and flatten
    acc = accuracy_score(target_np, pred_np)     # Compute pixel-wise accuracy
    dice = f1_score(target_np, pred_np, zero_division=0)     # Compute Dice score (F1 score for segmentation)
    iou = jaccard_score(target_np, pred_np, zero_division=0)     # Compute Intersection-over-Union (IoU)
    return dice, iou, acc     # Return all three metrics

# LOAD TEST DATA
with open(PICKLE_PATH, "rb") as f:     # Open pickle file in binary read mode
    data = pickle.load(f)     # Load stored tensors from file

images = data["test_images"]# Extract test images tensor
masks = data["test_masks"] # Extract test masks tensor

dataset = TensorDataset(images, masks) # Create dataset using images and masks
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False) # Create DataLoader for batching test data (no shuffle for evaluation)


# LOAD TRAINED MODEL 
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE) # Load saved model checkpoint to CPU/GPU
model = EfficientUNet().to(DEVICE) # Initialize model architecture and move to device
model.load_state_dict(checkpoint["model_state_dict"]) # Load trained weights into the model
model.eval() # Set model to evaluation mode (disables dropout, fixes batch norm)

# EVALUATION LOOP
total_acc = total_dice = total_iou = 0.0 # Variables to accumulate metrics
steps = 0 # Counter for number of batches processed

with torch.no_grad():     # Disable gradient computation (faster and memory efficient)
    for imgs, masks in dataloader:         # Loop through batches of test data
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)         # Move batch to CPU/GPU
        preds = model(imgs)         # Forward pass to get predictions
        dice, iou, acc = segmentation_metrics(preds, masks)         # Compute segmentation metrics
        total_acc += acc         # Accumulate accuracy
        total_dice += dice        # Accumulate Dice score
        total_iou += iou        # Accumulate IoU score
        steps += 1        # Increment batch counter

# PRINT FINAL RESULTS
print("Evaluation Results:")# Print evaluation header
print(f"Acc:   {total_acc/steps:.4f}")# Print average accuracy
print(f"Dice:  {total_dice/steps:.4f}")# Print average Dice score
print(f"IoU:   {total_iou/steps:.4f}")# Print average IoU score

# VISUALIZATION FUNCTION
def visualize_samples(model, dataloader, num_samples=5):    # Function to visualize model predictions vs ground truth

    model.eval()    # Ensure model is in evaluation mode
    shown = 0    # Counter for number of samples shown

    with torch.no_grad():        # Disable gradient computation
        for imgs, masks in dataloader:            # Loop through batches
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)            # Move data to CPU/GPU
            preds = (torch.sigmoid(model(imgs)) > THRESHOLD).float()            # Generate binary predictions
            for i in range(len(imgs)):                # Loop through images in batch
                img_np = imgs[i].cpu().permute(1, 2, 0).numpy()                # Convert image tensor to NumPy format for plotting
                mask_np = masks[i, 0].cpu().numpy()                # Extract ground truth mask
                pred_np = preds[i, 0].cpu().numpy()                # Extract predicted mask

                plt.figure(figsize=(12, 4))                # Create a figure for visualization
                plt.subplot(1, 3, 1)                # First subplot: original image
                plt.imshow(img_np, cmap="gray")               
                plt.title("Original MRI")
                plt.axis("off")

                plt.subplot(1, 3, 2)                # Second subplot: ground truth mask
                plt.imshow(mask_np, cmap="gray")
                plt.title("Ground Truth Mask")
                plt.axis("off")

                plt.subplot(1, 3, 3) # Third subplot: predicted mask
                plt.imshow(pred_np, cmap="gray")
                plt.title("Predicted Mask")
                plt.axis("off")

                plt.tight_layout()                # Adjust layout spacing
                plt.show()                # Display the plots
                shown += 1                # Increment displayed sample count
                if shown >= num_samples:                    # Stop once required number of samples are shown
                    return

# RUN VISUALIZATION
visualize_samples(model, dataloader, NUM_VISUALS) # Visualize predictions on test data
