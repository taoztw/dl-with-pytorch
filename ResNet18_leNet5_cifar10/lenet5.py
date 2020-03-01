import torch
from torch import nn


class Lenet5(nn.Module):
    """
    for cifar10 dataset.
    """

    def __init__(self):
        super(Lenet5, self).__init__()

        self.conv_unit = nn.Sequential(
            # x: [b,3,32,32] => [b,6,]
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            #
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        )

        # flatten
        # fc unit
        self.fc_unit = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

        # [b,3,32,32]
        # tmp = torch.randn(2, 3, 32, 32)
        # out = self.conv_unit(tmp)
        # # [b,16,5,5]
        # print('conv_out:', out.shape)

    def forward(self, x):
        """

        :param x: [b,3,32,32]
        :return:
        """
        batch_size = x.size(0)
        # [b,3,32,32] => [b,16,5,5]
        x = self.conv_unit(x)
        # [b,16,5,5] => [b,16*5*5]
        x = x.view(batch_size, 16*5*5)
        # [b,16*5*5] => [b,10]
        logits = self.fc_unit(x)

        return logits


def main():
    net = Lenet5()

    tmp = torch.randn(2, 3, 32, 32)
    out = net(tmp)
    print('lenet5 out:', out.shape)


if __name__ == '__main__':
    main()
