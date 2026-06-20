import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

# Import your perfectly working dataset and model
from dataset25 import DolphinFusionDataset25
from model25_attention import OrcaFusionNet25Attention

DATA_PATH = "/home/uq22yhaf/click_spot_sir_project/data/new_data/audio/"

# =======================================================
# NEW SAVE PATH: Updated to avoid overwriting old tests
# =======================================================
SAVE_PATH_BEST = "../models/orca_FOD_25feat_attention_focal_FINAL.pth"

BATCH_SIZE = 8
EPOCHS = 80  
LEARNING_RATE = 1e-4

# =======================================================
# CUSTOM WEIGHTED FOCAL LOSS
# =======================================================
class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.5): 
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha 
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Calculate standard Cross Entropy Loss using your specific dataset weights
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        
        # Get the probability of the correct class
        pt = torch.exp(-ce_loss)
        
        # Apply the Focal Loss formula
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        return focal_loss.mean()

def train():
    os.makedirs("../models", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the fixed dataset
    base_dataset = DolphinFusionDataset25(DATA_PATH, mode='val')
    labels_array = np.array(base_dataset.labels).astype(int)
    
    num_classes = 3
    
    # Using your exact calculated weights for the Alpha parameter
    dataset_weights = torch.tensor([2.5054, 2.6039, 0.4511], dtype=torch.float32).to(device)

    print("Performing Stratified Split...")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(np.zeros(len(labels_array)), labels_array))

    train_dataset = Subset(DolphinFusionDataset25(DATA_PATH, mode='train'), train_idx)
    val_dataset = Subset(base_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # LOAD THE ATTENTION MODEL
    model = OrcaFusionNet25Attention(num_classes=num_classes).to(device)
    
    # Pure Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Initialize the new Loss Function
    criterion = WeightedFocalLoss(alpha=dataset_weights, gamma=1.5)

    best_f1 = 0.0

    print("==========================================================")
    print(f"Starting 25-Feat ATTENTION + WEIGHTED FOCAL LOSS")
    print(f"Model will be saved to: {SAVE_PATH_BEST}")
    print("==========================================================")

    for epoch in range(EPOCHS):
        model.train()
        train_losses = []
        for images, meta, labels in train_loader:
            images, meta, labels = images.to(device), meta.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, meta)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, meta, labels in val_loader:
                images, meta, labels = images.to(device), meta.to(device), labels.to(device)
                outputs = model(images, meta)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"Epoch [{epoch+1:03d}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Macro F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), SAVE_PATH_BEST)
            print(f"--> Saved Best Model! (New Best F1: {best_f1:.4f})")

if __name__ == "__main__":
    train()