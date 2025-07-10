import torch
from model.cnn_doodle import DoodleCNN
from torchvision import transforms
from PIL import Image
import os

def predict_image(image_path, model_path="saved_models/best_model.pth", class_names=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DoodleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    if class_names:
        predicted_class = class_names[predicted.item()]
        print(f"Tahmin Edilen Sınıf: {predicted_class}")
    else:
        print(f"Tahmin Edilen Sınıf Index'i: {predicted.item()}")

    return predicted.item()
