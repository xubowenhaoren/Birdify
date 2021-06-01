import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from cnn_finetune import make_model
import os

# from google.colab import drive
# drive.mount('/content/gdrive')

# import os
# !pip install efficientnet_pytorch
# !pip install cnn_finetune

image_dimension = 512
batch_size = 16
num_workers = 4
num_classes = 555
k_fold_number = 10
run_k_fold_times = 4
folder_location = "/content/gdrive/MyDrive/kaggle/"
model_type = "inception"


def get_bird_data(augmentation=0):
    model = make_model(
        "inception_v4",
        pretrained=True,
        num_classes=num_classes,
        dropout_p=0.2,
        input_size=(image_dimension, image_dimension)
    )
    model = model.to(device)
    transform_train = transforms.Compose([
        transforms.Resize(image_dimension),
        transforms.RandomCrop(image_dimension, padding=8, padding_mode='edge'),  # Take 128x128 crops from padded images
        transforms.RandomHorizontalFlip(),  # 50% of time flip image along y-axis
        transforms.ToTensor(),
        transforms.Normalize(
            mean=model.original_model_info.mean,
            std=model.original_model_info.std)
    ])

    transform_test = transforms.Compose([
        transforms.Resize(image_dimension),
        transforms.CenterCrop(image_dimension),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=model.original_model_info.mean,
            std=model.original_model_info.std)
    ])
    trainset = torchvision.datasets.ImageFolder(root=folder_location+'train', transform=transform_train)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testset = torchvision.datasets.ImageFolder(root=folder_location+'test', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=num_workers)

    classes = open(folder_location+"names.txt").read().strip().split("\n")

    # Backward mapping to original class ids (from folder names) and species name (from names.txt)
    class_to_idx = trainset.class_to_idx
    idx_to_class = {int(v): int(k) for k, v in class_to_idx.items()}
    idx_to_name = {k: classes[v] for k, v in idx_to_class.items()}
    return model, {'dataset': trainset, 'train': trainloader, 'test': testloader, 'to_class': idx_to_class, 'to_name': idx_to_name}


def train(net, dataloader, epochs, optimizer, scheduler, k_fold_idx, run_idx):
    net.to(device)
    net.train()
    criterion = nn.CrossEntropyLoss()
    effective_epoch = (run_idx * k_fold_number) + k_fold_idx
    schedule = {0: 0.09, 5: 0.01, 15: 0.001, 20: 0.0001, 30: 0.00001}
    if effective_epoch in schedule:
        print("about to use learning rate", schedule[effective_epoch])
        for g in optimizer.param_groups:
            g['lr'] = schedule[effective_epoch]
    acc = 0.0
    for epoch in range(epochs):
        progress_bar = tqdm(enumerate(dataloader))
        for i, batch in progress_bar:
            inputs, labels = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()  # autograd magic, computes all the partial derivatives
            optimizer.step()  # takes a step in gradient direction

            get_acc_limit = 100
            if i % get_acc_limit == get_acc_limit - 1:
                # see predicted result
                softmax = torch.exp(outputs).cpu()
                prob = list(softmax.detach().numpy())
                predictions = np.argmax(prob, axis=1)
                acc = accuracy(predictions, batch[1].numpy())
            progress_bar.set_description(str(("epoch", effective_epoch, "lr", scheduler.get_lr(), "acc", acc, "loss", loss.item())))

        scheduler.step()


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


def accuracy(y_pred, y):
    return np.sum(y_pred == y).item()/y.shape[0]


# define a cross validation function
def cross_valid(model, dataset, k_fold, times):
    total_size = len(dataset)
    fraction = 1 / k_fold
    seg = int(total_size * fraction)
    checkpoint_path = folder_location + model_type + '.pth'
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.975)
    if os.path.exists(checkpoint_path):
        print("found checkpoint, recovering")
        checkpoint = torch.load(checkpoint_path)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler = checkpoint['scheduler']
        model.load_state_dict(checkpoint['model'])
    else:
        print("no checkpoint, using new optimizer and scheduler")
    for i in range(k_fold):
        trll = 0
        trlr = i * seg
        vall = trlr
        valr = i * seg + seg
        trrl = valr
        trrr = total_size

        train_left_indices = list(range(trll, trlr))
        train_right_indices = list(range(trrl, trrr))

        train_indices = train_left_indices + train_right_indices
        val_indices = list(range(vall, valr))

        train_set = torch.utils.data.dataset.Subset(dataset, train_indices)
        val_set = torch.utils.data.dataset.Subset(dataset, val_indices)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                   shuffle=True, num_workers=num_workers)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                                 shuffle=True, num_workers=num_workers)
        train(model, train_loader, 1, optimizer, scheduler, i, times)
        train(model, val_loader, 1, optimizer, scheduler, i, times)
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler
        }
        torch.save(checkpoint, checkpoint_path)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device", device)
    print("The total effective training epochs are", run_k_fold_times * k_fold_number)
    model, data = get_bird_data()

    for idx in range(run_k_fold_times):
        cross_valid(model, data['dataset'], k_fold_number, idx)
    predict_file_path = folder_location + model_type + ".csv"
    predict(model, data['test'], predict_file_path)
