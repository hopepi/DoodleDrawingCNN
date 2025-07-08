from torchvision import transforms,datasets
from torch.utils.data import DataLoader


def get_dataloaders(train_dir,test_dir,batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir,transform=transform)

    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

    return train_loader, test_loader