import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch import nn, optim
from lenet5 import Lenet5
from resnet import ResNet18


def main():
    batch_size = 32

    cifar_train = datasets.CIFAR10('cifar', train=True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batch_size, shuffle=True)

    cifar_test = datasets.CIFAR10('cifar', train=False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batch_size, shuffle=True)

    x, label = iter(cifar_train).next()
    print('x:', x.shape, 'label:', label.shape)

    # nvidia-smi
    # device = torch.device('cuda')
    # model = Lenet5().to(device)
    # model = Lenet5()
    model = ResNet18()
    # criteon = nn.CrossEntropyLoss().to(device)
    criteon = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(model)

    for epoch in range(1000):

        model.train()
        for batchidx, (x, label) in enumerate(cifar_train):
            # [b,3,32,32]
            # [b]
            # x, label = x.to(device), label.to(device)
            x, label = x, label
            logits = model(x)
            # logits : [b,10]
            # label :[b]
            loss = criteon(logits, label)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch, loss.item())

        model.eval()
        with torch.no_grad():
            # test
            total_correct = 0
            total_num = 0
            for x, label in cifar_test:
                # [b,3,32,32]
                # [b]
                # x, label = x.to(device),label.to(device)
                x, label = x, label
                # [b,10]
                logits = model(x)
                pred = logits.argmax(dim=1)
                total_correct += torch.eq(pred, label).float().sum().item()
                total_num += x.size(0)

            acc = total_correct / total_num
            print(epoch, acc)


if __name__ == '__main__':
    main()
