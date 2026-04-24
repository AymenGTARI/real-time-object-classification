import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# =====================
# CONFIG
# =====================
DATA_DIR   = "dataset"
BATCH_SIZE = 16
EPOCHS     = 30
LR         = 0.0001
PATIENCE   = 7
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation de : {DEVICE}")

# =====================
# TRANSFORMS
# =====================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =====================
# DATASET
# =====================
full_dataset = datasets.ImageFolder(DATA_DIR)
class_names  = full_dataset.classes
print(f"Classes ({len(class_names)}) : {class_names}")

train_size = int(0.8 * len(full_dataset))
val_size   = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Transforms séparés train / val
train_dataset.dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
val_dataset.dataset   = datasets.ImageFolder(DATA_DIR, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# =====================
# MODEL
# =====================
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

# Dégel partiel : layer3, layer4 et fc
for name, param in model.named_parameters():
    param.requires_grad = any(k in name for k in ("layer3", "layer4", "fc"))

model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.fc.in_features, len(class_names))
)
model = model.to(DEVICE)

criterion        = nn.CrossEntropyLoss()
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer        = optim.Adam(trainable_params, lr=LR, weight_decay=1e-4)
scheduler        = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

# =====================
# ENTRAÎNEMENT
# =====================
best_acc         = 0.0
patience_counter = 0
prev_lr          = LR
train_losses     = []
val_accuracies   = []
train_accuracies = []

for epoch in range(EPOCHS):
    current_lr = optimizer.param_groups[0]['lr']
    if current_lr != prev_lr:
        print(f"  📉 LR réduit : {prev_lr:.6f} → {current_lr:.6f}")
        prev_lr = current_lr

    print(f"\n{'='*50}")
    print(f"Epoch {epoch+1}/{EPOCHS}  |  LR: {current_lr:.6f}")
    print(f"{'='*50}")

    # -- Train --
    model.train()
    running_loss  = 0.0
    correct_train = 0
    total_train   = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss  += loss.item()
        _, preds       = torch.max(outputs, 1)
        correct_train += (preds == labels).sum().item()
        total_train   += labels.size(0)

    epoch_loss      = running_loss / len(train_loader)
    epoch_train_acc = correct_train / total_train
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_train_acc)

    # -- Validation --
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            _, preds = torch.max(model(images), 1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    val_acc = correct / total
    val_accuracies.append(val_acc)

    print(f"  Train Loss : {epoch_loss:.4f}  |  Train Acc : {epoch_train_acc:.4f}")
    print(f"  Val   Acc  : {val_acc:.4f}")

    scheduler.step(val_acc)

    # -- Early stopping --
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "model.pth")
        print(f"  ✅ Meilleur modèle sauvegardé (val_acc={best_acc:.4f})")
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"  ⏳ Pas d'amélioration ({patience_counter}/{PATIENCE})")
        if patience_counter >= PATIENCE:
            print("  ⛔ Early stopping déclenché")
            break

print(f"\n🏁 Entraînement terminé — Meilleure val accuracy : {best_acc:.4f}")

# =====================
# GRAPHIQUES
# =====================
epochs_range = range(1, len(train_losses) + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Résultats d'entraînement", fontsize=14)

ax1.plot(epochs_range, train_losses, color="tomato", label="Train Loss")
ax1.set_title("Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(epochs_range, train_accuracies, color="steelblue", label="Train Acc")
ax2.plot(epochs_range, val_accuracies,   color="seagreen",  label="Val Acc")
ax2.set_title("Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_ylim(0, 1)
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_results.png", dpi=150)
plt.show()
print("📊 Graphique sauvegardé : training_results.png")