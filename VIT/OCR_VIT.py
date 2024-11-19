import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from transformers import ViTForImageClassification
from torchvision import transforms
from PIL import Image
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set up the data directory and transformation (adjust as per your data location)
data_dir = '../dataset/'  # Update to your dataset directory
batch_size = 4  # Reduced batch size to save memory

# Transformation pipeline for the images
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# Load data
def load_data(data_dir):
    img_paths = []
    labels = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for img_file in glob.glob(os.path.join(label_dir, '*.png')):
                img_paths.append(img_file)
                try:
                    labels.append(int(label))
                except ValueError:
                    labels.append(ord(label) - ord('A') + 10)
    return img_paths, labels

img_paths, labels = load_data(data_dir)
dataset = CustomDataset(img_paths, labels, transform=transform)

# Split the data into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Load the Vision Transformer Model (use smaller variant to save memory if needed)
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=len(set(labels)),  # Unique number of labels
    ignore_mismatched_sizes=True ).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
scaler = torch.cuda.amp.GradScaler()  # Enable mixed precision

# Training and validation loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    torch.cuda.empty_cache()

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images).logits
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Validation accuracy
    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy:.2f}%')

# Classification report
report = classification_report(y_true, y_pred)
print(report)

# Confusion Matrix Visualization
confusion_mat = confusion_matrix(y_true, y_pred)
cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_mat)
cm_display.plot()
plt.title('Confusion Matrix')
plt.show()

# Save the model
model_path = "vit_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")