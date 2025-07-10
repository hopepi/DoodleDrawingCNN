import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from model.cnn_doodle import DoodleCNN
from utils.dataloader import get_dataloaders
import os
import math

def evaluate_per_class(model_path="../saved_models/best_model.pth", classes_per_page=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DoodleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    _, test_loader = get_dataloaders(
        "../data/doodle_split/train",
        "../data/doodle_split/test",
        batch_size=64,
    )

    class_names = test_loader.dataset.classes
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            for label, pred in zip(labels, predicted):
                class_total[label.item()] += 1
                if label == pred:
                    class_correct[label.item()] += 1

    class_accuracies = {}
    for i, class_name in enumerate(class_names):
        total = class_total[i]
        correct = class_correct[i]
        acc = 100 * correct / total if total > 0 else 0
        class_accuracies[class_name] = acc

    sorted_accuracies = sorted(class_accuracies.items(), key=lambda x: x[1])

    total_classes = len(sorted_accuracies)
    total_pages = math.ceil(total_classes / classes_per_page)

    os.makedirs("../results/evaluateResult", exist_ok=True)

    for page in range(total_pages):
        start_idx = page * classes_per_page
        end_idx = start_idx + classes_per_page
        page_data = sorted_accuracies[start_idx:end_idx]

        plt.figure(figsize=(10, 6))
        plt.barh([x[0] for x in page_data], [x[1] for x in page_data], color='skyblue')
        plt.xlabel("Başarı (%)")
        plt.title(f"Sınıf Bazlı Test Doğrulukları (Sayfa {page + 1}/{total_pages})")
        plt.grid(axis='x')
        plt.tight_layout()

        save_path = f"../results/evaluateResult/per_class_accuracy_{page + 1}.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Kaydedildi: {save_path}")

    weakest = sorted_accuracies[:15]
    strongest = sorted_accuracies[-15:]

    result_txt_path = "../results/evaluateTxt/class_accuracy_summary.txt"
    with open(result_txt_path, "w", encoding="utf-8") as f:
        print("\nEn zayıf 15 sınıf:")
        f.write("En zayıf 15 sınıf:\n")
        for name, acc in weakest:
            line = f"{name:<25}: {acc:.2f}%"
            print(line)
            f.write(line + "\n")

        print("\nEn iyi 15 sınıf:")
        f.write("\nEn iyi 15 sınıf:\n")
        for name, acc in strongest:
            line = f"{name:<25}: {acc:.2f}%"
            print(line)
            f.write(line + "\n")

    print(f"\nÖzet dosyası kaydedildi: {result_txt_path}")


if __name__ == "__main__":
    evaluate_per_class()