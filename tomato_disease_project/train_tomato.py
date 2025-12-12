import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Using Apple Silicon GPU (MPS) if available
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", DEVICE)

DATA_DIR = "data"
NUM_CLASSES = 11
BATCH_SIZE = 32
EPOCHS = 10  # Faster training first round
LR = 1e-4
MODEL_PATH = "models/best_model.pth"
IMG_SIZE = 160  # Reduced resolution => fast

def get_dataloaders():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8,1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    valid_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
    valid_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "valid"), transform=valid_tf)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_dl, valid_dl, train_ds.classes


def train():
    os.makedirs("models", exist_ok=True)
    train_dl, valid_dl, classes = get_dataloaders()

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model.to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

        # 📌 Validation + Metrics
        model.eval()
        true_labels = []
        pred_labels = []

        with torch.no_grad():
            for x, y in valid_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                _, preds = torch.max(out, 1)
                true_labels.extend(y.cpu().numpy())
                pred_labels.extend(preds.cpu().numpy())

        # 📌 Compute Metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels, average="weighted", zero_division=0)
        recall = recall_score(true_labels, pred_labels, average="weighted", zero_division=0)
        f1 = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)

        print(f"\n📍 Epoch {epoch+1}/{EPOCHS}")
        print(f"✔ Accuracy:  {accuracy:.4f}")
        print(f"✔ Precision: {precision:.4f}")
        print(f"✔ Recall:    {recall:.4f}")
        print(f"✔ F1 Score:  {f1:.4f}")

        # 📌 Save Best Model based on accuracy
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save({"model": model.state_dict(), "classes": classes}, MODEL_PATH)
            print("🔥 Best model saved ✔")

    print("\n🎯 Training Completed!")
    print("✨ Best Validation Accuracy:", round(best_acc, 4))


if __name__ == "__main__":
    train()
