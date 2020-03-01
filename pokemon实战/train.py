from pokemon import Pokemon
from resnet import ResNet18

from torch.utils.data import DataLoader
from torch import optim, nn
# import visdom
import torch

batch_sz = 32
lr = 1e-3
epochs = 10

device = torch.device('cuda')
torch.manual_seed(1234)

train_db = Pokemon('pokeman', 224, mode='train')
val_db = Pokemon('pokeman', 224, mode='val')
test_db = Pokemon('pokeman', 224, mode='test')

train_loader = DataLoader(train_db, batch_size=batch_sz, shuffle=True)
val_loader = DataLoader(val_db, batch_size=batch_sz)
test_loader = DataLoader(test_db, batch_size=batch_sz)


def evalute(model, loader):
    model.eval()

    correct = 0
    total = len(loader.dataset)
    # print(total)

    for x, y in loader:
        # .to(device)
        x,y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()

    return correct / total


def main():
    model = ResNet18(5).to(device)  # .to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc, best_epoch = 0, 0

    for epoch in range(epochs):

        for step, (x, y) in enumerate(train_loader):
            # x: [b,3,224,224] , y: [b]
            # x.to(device), y.to(device)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



        print(epoch, loss.item())

        # validation
        if epoch % 1 == 0:

            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:

                best_epoch = epoch
                print('best_epoch, ', best_epoch)
                best_acc = val_acc
                print('best_acc,  ', best_acc)

                torch.save(model.state_dict(), 'best.mdl')

    print('best acc: ', best_acc, 'best epoch: ', best_epoch)

    model.load_state_dict(torch.load('best.mdl'))
    print('loaded from ckpt')

    test_acc = evalute(model, test_loader)
    print('test_acc: ', test_acc)


if __name__ == '__main__':
    main()
