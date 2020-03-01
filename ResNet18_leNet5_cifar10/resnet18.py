import torch
from torch import nn
import torch.nn.functional as F


class ResBlk(nn.Module):
    """
    resnet block
    """

    def __init__(self, ch_in, ch_out):
        """

        :param ch_in:
        :param ch_out:
        """
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            # [b,ch_in,h,w] => [b,ch_out,h,w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=2),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # print('out.shape', out.shape)
        test = self.extra(x)
        # print('extra shape', test.shape)
        # short cut
        out = self.extra(x) + out

        return out


class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(64)
        )

        # followed 4 blocks
        # [b,64,h,w] => [b,128,h,w]
        self.blk1 = ResBlk(64, 128)
        # [b,128,h,w] => [b,256,h,w]
        self.blk2 = ResBlk(128, 256)
        # [b,256,h,w] => [b,512,h,w]
        self.blk3 = ResBlk(256, 512)
        self.blk4 = ResBlk(512, 512)

        self.outlayer = nn.Linear(512 * 1 * 1, 10)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = F.relu(self.conv1(x))
        # print('conv1 shape', x.shape)
        # [b,64,h,w] => [b,1024,h,w]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        # print('after conv:', x.shape)  # [b,512,]
        # [b,512,h,w] => [b,512,1,1]
        x = F.adaptive_avg_pool2d(x, [1, 1])
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)

        return x


def main():
    x = torch.randn(2, 3, 32, 32)
    model = ResNet18()
    out = model(x)
    print('resnet', out.shape)


if __name__ == '__main__':
    main()
