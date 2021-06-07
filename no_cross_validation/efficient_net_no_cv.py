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
num_epochs = 20
folder_location = "/content/gdrive/MyDrive/kaggle/"
model_type = "efficient_net_no_cv_weight_decay"
dropout_p = 0.2
per_epoch_lr_decay = 0.9
recovered = False
use_dropout = False
weight_decay = 0.0005


class MyEfficientNet(nn.Module):

    def __init__(self):
        super().__init__()

        # EfficientNet
        self.network = EfficientNet.from_pretrained('efficientnet-b5', num_classes=num_classes)

        # Replace last layer
        self.network._fc = nn.Sequential(nn.Linear(self.network._fc.in_features, 512),
                                         nn.ReLU(),
                                         nn.Dropout(dropout_p),
                                         nn.Linear(512, num_classes))

    def forward(self, x):
        out = self.network(x)
        return out


def get_bird_data(augmentation=0):
    if use_dropout:
        model = MyEfficientNet()
    else:
        model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=num_classes)
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


def train(net, dataloader, epochs, optimizer):
    net.to(device)
    net.train()
    criterion = nn.CrossEntropyLoss()
    acc = 0.0
    loss_sum = 0.0
    total_itr = 0
    for epoch in range(start_epoch, epochs):
        schedule = {0: 0.09, 5: 0.01, 15: 0.001, 20: 0.0001, 30: 0.00001}
        if epoch in schedule:
            print("found new schedule, overriding optimizer learning rate")
            new_lr = schedule[epoch]
        else:
            new_lr = optimizer.param_groups[0]['lr'] * per_epoch_lr_decay
        for g in optimizer.param_groups:
            g['lr'] = new_lr

        progress_bar = tqdm(enumerate(dataloader))
        for i, batch in progress_bar:
            inputs, labels = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()  # autograd magic, computes all the partial derivatives
            curr_loss = loss.item()
            loss_sum += curr_loss
            total_itr += 1
            optimizer.step()  # takes a step in gradient direction

            get_acc_limit = 100
            if i % get_acc_limit == get_acc_limit - 1:
                # see predicted result
                softmax = torch.exp(outputs).cpu()
                prob = list(softmax.detach().numpy())
                predictions = np.argmax(prob, axis=1)
                acc = accuracy(predictions, batch[1].numpy())
            progress_bar.set_description(str(("epoch", epoch, "lr", optimizer.param_groups[0]['lr'], "acc", acc, "loss", curr_loss)))
        train_acc, train_loss = str(acc), str(loss_sum / total_itr)
        checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch
                    }
        # So we write the log before the checkpoint. If we get duplicate epochs, use the last line.
        with open(folder_location + model_type + "_log.txt", "a+") as log_file:
            log_file.write("epoch " + str(epoch) + " train acc " + train_acc + " loss " + train_loss + "\n")
        torch.save(checkpoint, checkpoint_path)


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


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device", device)
    print("The total training epochs are", num_epochs)
    model, data = get_bird_data()
    checkpoint_path = folder_location + model_type + '.pth'
    if weight_decay > 0:
        print("using weight decay", weight_decay)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=weight_decay)
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print("found checkpoint, recovering")
        checkpoint = torch.load(checkpoint_path)
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        print("no checkpoint, using new optimizer")
    train(model, data["train"], num_epochs, optimizer)
    predict_file_path = folder_location + model_type + ".csv"
    predict(model, data['test'], predict_file_path)
