import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
import wandb
import eval_cifar100
import eval_ood

# ------------------ Configuration ------------------
# Contains training settings, device type, data paths, and WandB setup
CONFIG = {
    "model": "resnet18",  # Name of the model used
    "batch_size": 32,
    "learning_rate": 0.0005,
    "epochs": 15,
    "num_workers": 4,
    "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
    "data_dir": "./data",          # Path for CIFAR-100 data
    "ood_dir": "./data/ood-test",  # Path for OOD test data
    "wandb_project": "sp25-ds542-challenge",
    "seed": 42,
}

# ------------------ Data Preparation ------------------

# Training transforms with data augmentation to improve generalization
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Validation and test transforms (no augmentation)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load full training dataset and split into training/validation
trainset_full = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=True, download=True, transform=transform_train)
train_size = int(0.8 * len(trainset_full))
val_size = len(trainset_full) - train_size
trainset, valset = random_split(trainset_full, [train_size, val_size])

# Loaders for training, validation, and testing
trainloader = DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
valloader = DataLoader(valset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
testset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

# ------------------ Model Definition ------------------

from torchvision.models import resnet18

# Load ResNet18 without pretrained weights
model = resnet18(weights=None)

# Replace final classification head with 100-class output
model.fc = nn.Linear(model.fc.in_features, 100)

# Move model to GPU/CPU
model = model.to(CONFIG["device"])

# ------------------ Training Function ------------------
def train(epoch, model, trainloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Track progress bar per batch
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)

    for i, (inputs, labels) in enumerate(progress_bar):
        inputs = inputs.to(CONFIG["device"])
        labels = labels.to(CONFIG["device"])

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({"loss": running_loss / (i+1), "acc": 100. * correct / total})

    return running_loss / len(trainloader), 100. * correct / total

# ------------------ Validation Function ------------------
def validate(model, valloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)
        for i, (inputs, labels) in enumerate(progress_bar):
            inputs = inputs.to(CONFIG["device"])
            labels = labels.to(CONFIG["device"])

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})

    return running_loss / len(valloader), 100. * correct / total

# ------------------ Main Training Loop ------------------
def main():
    wandb.init(project=CONFIG["wandb_project"], config=CONFIG)
    wandb.watch(model)

    # Loss, optimizer, and LR scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_val_acc = 0.0

    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion)
        val_loss, val_acc = validate(model, valloader, criterion)

        # Log training progress to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"]
        })

        scheduler.step()

        # Save best model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model_part2.pth")
            wandb.save("best_model_part2.pth")

    wandb.finish()

    # ------------------ Evaluation ------------------

    print("Evaluating clean test set...")
    model.load_state_dict(torch.load("best_model_part2.pth"))
    model.eval()
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

    print("Evaluating OOD test set...")
    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)
    submission_df = eval_ood.create_ood_df(all_predictions)
    submission_df.to_csv("submission_ood_part2.csv", index=False)
    print("✅ submission_ood_part2.csv created!")

# ------------------ Entry Point ------------------
if __name__ == "__main__":
    main()
