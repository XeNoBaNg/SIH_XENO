# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import time

# Import your simplified backbone model
from models.backbones.hybrid_vit_swin import HybridViTSwinBackbone

# --- 1. CONFIGURATION ---
DATA_DIR = "dataset"
MODEL_SAVE_PATH = "weights/backbone.pth"
NUM_CLASSES = 4  # Adjust this to the number of folders in your dataset/train directory
BATCH_SIZE = 8  # Lower this if you run out of memory (e.g., 8, 4)
NUM_EPOCHS = 50  # Number of times to loop through the entire dataset
LEARNING_RATE = 0.0001

def train_model():
    # --- 2. SETUP DEVICE, DATA TRANSFORMS, AND DATALOADERS ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data augmentation for the training set to make the model more robust
    # Data normalization for both training and validation sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Loading datasets...")
    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    # Verify that the number of detected classes matches NUM_CLASSES
    if len(class_names) != NUM_CLASSES:
        raise ValueError(f"Discrepancy: NUM_CLASSES is {NUM_CLASSES} but found {len(class_names)} folders in {os.path.join(DATA_DIR, 'train')}. Please ensure they match.")
    print(f"Found {len(class_names)} classes: {', '.join(class_names)}")


    # --- 3. INITIALIZE THE MODEL, LOSS FUNCTION, AND OPTIMIZER ---
    print("Initializing model...")
    # We use the simplified Swin-only backbone for stability
    model = HybridViTSwinBackbone(num_classes=NUM_CLASSES, device=device)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # --- 4. TRAINING AND VALIDATION LOOP ---
    since = time.time()
    best_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Save the model if it has the best validation accuracy so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # Ensure the 'weights' directory exists
                os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"New best model saved to {MODEL_SAVE_PATH} with accuracy: {best_acc:.4f}")

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

if __name__ == '__main__':
    train_model()