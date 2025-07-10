import torch
import torch.nn as nn
import torch.optim as optim
import os
from model.cnn_doodle import DoodleCNN
from utils.dataloader import get_dataloaders
from tqdm import tqdm

def train_model(epochs=40, batch_size=64, lr=0.001,
                model_path="saved_models", checkpoint_path="saved_models/checkpoint.pth"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DoodleCNN().to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    start_epoch = 1
    best_accuracy = 0.0
    epochs_no_improve = 0
    early_stop_patience = 5

    if os.path.exists(checkpoint_path):
        print("Checkpoint bulundu. Model eğitime kaldığı yerden devam edecek...\n")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
    else:
        print("Checkpoint bulunamadı. Eğitime sıfırdan başlanıyor...\n")

    train_loader, test_loader = get_dataloaders(
        "data/doodle_split/train",
        "data/doodle_split/test",
        batch_size=batch_size,
    )

    os.makedirs(model_path, exist_ok=True)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loop.set_postfix(loss=loss.item())

        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)

        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_acc = 100 * test_correct / test_total

        print(f"\nEpoch [{epoch}/{epochs}] ---- Loss: {avg_loss:.4f} ---- Train Acc: {train_acc:.4f}% ---- Test Acc: {test_acc:.4f}%")

        scheduler.step(test_acc)

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss
        }, checkpoint_path)

        torch.save(model.state_dict(), os.path.join(model_path, f"model_epoch_{epoch}.pth"))

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(model_path, "best_model.pth"))
            print(f"Yeni en iyi model kaydedildi: {best_accuracy:.2f}%")
        else:
            epochs_no_improve += 1
            print(f"Gelişmeyen epoch sayısı: {epochs_no_improve}/{early_stop_patience}")

        # Early stopping kontrolü
        if epochs_no_improve >= early_stop_patience:
            print(f"Eğitim erken durduruldu. Test doğruluğu {early_stop_patience} epoch boyunca gelişmedi.")
            break
