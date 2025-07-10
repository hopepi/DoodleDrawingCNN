import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model.cnn_doodle import DoodleCNN
from utils.dataloader import get_dataloaders

def visualize_diverse_predictions(model_path="saved_models/best_model.pth", max_classes=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DoodleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    _, test_loader = get_dataloaders(
        "data/doodle_split/train",
        "data/doodle_split/test",
        batch_size=64,
    )

    class_names = test_loader.dataset.classes

    # Her sınıf için bir örnek topla
    selected_images = {}
    for images, labels in test_loader:
        for img, label in zip(images, labels):
            class_name = class_names[label.item()]
            if class_name not in selected_images:
                selected_images[class_name] = (img, label.item())
            if len(selected_images) >= max_classes:
                break
        if len(selected_images) >= max_classes:
            break

    plt.figure(figsize=(15, 6))
    for i, (class_name, (img, true_label)) in enumerate(selected_images.items()):
        image = img.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            prob = F.softmax(output, dim=1)
            confidence = prob[0].max().item()
            predicted = prob.argmax(dim=1).item()

        image_np = img.squeeze().cpu().numpy()

        plt.subplot(2, 5, i + 1)
        plt.imshow(image_np, cmap='gray')
        plt.title(f"Gerçek: {class_name}", fontsize=10)
        plt.xlabel(f"Tahmin: {class_names[predicted]} ({confidence*100:.1f}%)", fontsize=9)
        plt.xticks([]), plt.yticks([])

    plt.tight_layout()
    plt.show()
