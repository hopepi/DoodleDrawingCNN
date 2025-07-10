import torch
from model.cnn_doodle import DoodleCNN
from utils.dataloader import get_dataloaders
import os

def test_model(checkpoint_path="saved_models/best_model.pth", batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DoodleCNN().to(device)

    if not os.path.exists(checkpoint_path):
        print("Checkpoint bulunamadı! Önce modeli eğitmelisin.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, test_loader = get_dataloaders(
        "data/doodle_split/train",
        "data/doodle_split/test",
        batch_size=batch_size
    )

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Doğruluğu: {accuracy:.2f}%")


