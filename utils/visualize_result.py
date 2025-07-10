"""
JUST İMAGE CREATE
"""

import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import random
from model.cnn_doodle import DoodleCNN
from utils.dataloader import get_dataloaders

def save_diverse_prediction_images(model_path="saved_models/best_model.pth", output_dir="results", sets=10, classes_per_set=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DoodleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    _, test_loader = get_dataloaders(
        "../data/doodle_split/train",
        "../data/doodle_split/test",
        batch_size=64,
    )

    class_names = test_loader.dataset.classes

    all_images, all_labels = [], []
    for images, labels in test_loader:
        all_images.extend(images)
        all_labels.extend(labels)
        if len(all_images) >= 10000:
            break

    for set_index in range(sets):
        used_classes = set()
        selected_data = []

        indices = list(range(len(all_images)))
        random.shuffle(indices)

        for idx in indices:
            label = all_labels[idx].item()
            if label not in used_classes:
                selected_data.append((all_images[idx], label))
                used_classes.add(label)
            if len(selected_data) == classes_per_set:
                break

        plt.figure(figsize=(15, 6))
        for i, (img, true_label) in enumerate(selected_data):
            image = img.unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image)
                prob = F.softmax(output, dim=1)
                confidence = prob[0].max().item()
                predicted = prob.argmax(dim=1).item()

            image_np = img.squeeze().cpu().numpy()

            plt.subplot(2, 5, i + 1)
            plt.imshow(image_np, cmap='gray')
            plt.title(f"Gerçek: {class_names[true_label]}", fontsize=10)
            plt.xlabel(f"Tahmin: {class_names[predicted]} ({confidence*100:.1f}%)", fontsize=9)
            plt.xticks([]), plt.yticks([])

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"diverse_preds_{set_index + 1}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Kaydedildi: {save_path}")


if __name__=="__main__":
    save_diverse_prediction_images(
        model_path="../saved_models/best_model.pth",
        output_dir="../results/diverseResult",
        sets=10,
        classes_per_set=10
    )