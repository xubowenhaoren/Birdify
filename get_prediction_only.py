import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
import os

# If in google colab, first run
# from google.colab import drive
# drive.mount('/content/gdrive')
# !pip install efficientnet_pytorch

image_dimension = 512
batch_size = 8
num_workers = 4
num_classes = 555
k_fold_number = 10
run_k_fold_times = 1
folder_location = ""
model_type = "efficient_net"


def get_bird_data(augmentation=0):
    model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=num_classes)
    model.fc = nn.Linear(512, num_classes)
    model = model.to(device)
    transform_train = transforms.Compose([
        transforms.Resize(image_dimension),
        transforms.RandomCrop(image_dimension, padding=8, padding_mode='edge'),  # Take 128x128 crops from padded images
        transforms.RandomHorizontalFlip(),  # 50% of time flip image along y-axis
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(image_dimension),
        transforms.CenterCrop(image_dimension),
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.ImageFolder(root=folder_location + 'train', transform=transform_train)
    testset = torchvision.datasets.ImageFolder(root=folder_location+'test', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=num_workers)

    classes = open(folder_location+"names.txt").read().strip().split("\n")

    # Backward mapping to original class ids (from folder names) and species name (from names.txt)
    class_to_idx = trainset.class_to_idx
    idx_to_class = {int(v): int(k) for k, v in class_to_idx.items()}
    idx_to_name = {k: classes[v] for k, v in idx_to_class.items()}
    return model, {'test': testloader, 'to_class': idx_to_class, 'to_name': idx_to_name}


def predict(net, dataloader, ofname):
    out = open(ofname, 'w')
    out.write("path,class\n")
    net.to(device)
    net.eval()
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(dataloader)):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            fname, _ = dataloader.dataset.samples[i]
            out.write("test/{},{}\n".format(fname.split('/')[-1], data['to_class'][predicted.item()]))
    out.close()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device", device)
    print("The total effective training epochs are", run_k_fold_times * k_fold_number)
    model, data = get_bird_data()
    predict_file_path = folder_location + model_type + ".csv"
    checkpoint_path = folder_location + model_type + '.pth'
    if os.path.exists(checkpoint_path):
        print("found checkpoint, recovering")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
    predict(model, data['test'], checkpoint_path)
