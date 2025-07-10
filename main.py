import os

from scripts.predict import predict_image
from scripts.train import train_model
from scripts.test import test_model
import multiprocessing
import torch

def menu():
    print("\nDoodle Sınıflandırma Sistemi")
    print("1 - Modeli Eğit")
    print("2 - Modeli Test Et")
    print("3 - Modeli Tahmin Et")
    print("0 - Çıkış")

def main():
    while True:
        menu()
        choice = input("Seçimin: ")

        if choice == "1":
            train_model()

        elif choice == "2":
            test_model()

        elif choice == "3":
            image_path = input("Tahmin edilecek görselin yolunu girin: ")
            class_names = os.listdir("data/doodle_split/train")
            class_names.sort()
            predict_image(image_path, class_names=class_names)


        elif choice == "0":
            print("Çıkılıyor")
            break
        else:
            print("Geçersiz seçim. Lütfen tekrar deneyin")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn',force=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    main()
