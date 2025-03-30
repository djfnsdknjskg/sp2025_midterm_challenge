# ---------- Import Required Libraries ----------
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
from torchvision.models import resnet50, ResNet50_Weights

# ---------- Configuration ----------
# Define key hyperparameters and runtime settings
CONFIG = {
    "model": "resnet50",
    "batch_size": 64,
    "learning_rate": 0.0001,
    "epochs": 50,
    "num_workers": 4,
    "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
    "wandb_project": "sp25-ds542-challenge",
    "ood_dir": "./data/ood-test",
    "data_dir": "./data"
}

# Improve CPU parallelism for data loading
torch.set_num_threads(6)
torch.set_num_interop_threads(6)

# ---------- Data Augmentation ----------
# Apply aggressive augmentation to improve generalization
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),                      # flip image horizontally
    transforms.RandomCrop(32, padding=4),                   # randomly crop with padding
    transforms.RandomRotation(10),                          # random rotation
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),              # jitter brightness/contrast/saturation/hue
    transforms.ToTensor(),                                  # convert to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # normalize to mean=0, std=1
    transforms.RandomErasing()                              # randomly erase patches
])

# No augmentation for validation/test sets
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ---------- Load and Split Data ----------
trainset_full = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=True, download=True, transform=transform_train)
train_size = int(0.9 * len(trainset_full))
val_size = len(trainset_full) - train_size
trainset, valset = random_split(trainset_full, [train_size, val_size])

trainloader = DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
valloader = DataLoader(valset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
testset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

# ---------- Load and Modify Pretrained Model ----------
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)

# Unfreeze all layers so we can fine-tune the entire model
for param in model.parameters():
    param.requires_grad = True

# Replace classification head with Dropout + Linear (100 classes)
model.fc = nn.Sequential(
    nn.Dropout(0.5),                          # add dropout for regularization
    nn.Linear(model.fc.in_features, 100)      # output layer for 100 classes
)

# Move model to device (GPU/CPU)
model = model.to(CONFIG["device"])

# ---------- Training Function ----------
def train(epoch, model, trainloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)
    for i, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(CONFIG["device"]), labels.to(CONFIG["device"])

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

# ---------- Validation Function ----------
def validate(model, valloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)
        for i, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(CONFIG["device"]), labels.to(CONFIG["device"])
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({"loss": running_loss / (i+1), "acc": 100. * correct / total})

    return running_loss / len(valloader), 100. * correct / total

# ---------- Main Execution ----------
def main():
    # Initialize wandb tracking
    wandb.init(project=CONFIG["wandb_project"], config=CONFIG)
    wandb.watch(model)

    # CrossEntropy loss with label smoothing (for regularization)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # AdamW optimizer (better regularization than Adam)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])

    # Cosine annealing scheduler (smooth learning rate decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    best_val_acc = 0.0
    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion)
        val_loss, val_acc = validate(model, valloader, criterion)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"]
        })

        scheduler.step()

        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model_part3.pth")
            wandb.save("best_model_part3.pth")

    wandb.finish()

    # ---------- Final Evaluation ----------
    print("Evaluating clean test set...")
    model.load_state_dict(torch.load("best_model_part3.pth"))
    model.eval()
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

    print("Evaluating OOD test set...")
    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)
    submission_df = eval_ood.create_ood_df(all_predictions)
    submission_df.to_csv("submission_ood_part3.csv", index=False)
    print("✅ submission_ood_part3.csv created!")

# ---------- Run ----------
if __name__ == "__main__":
    main()
