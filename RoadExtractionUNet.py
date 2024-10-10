import torch
import torch.nn as nn
import torchvision.transforms.functional as F

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, st=1):
        super(Block, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,stride=st, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

        self.identity = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=st),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x.clone()
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.conv2(x)
        x += self.identity(identity)
        return x
    
class ReUNet(nn.Module):
    def __init__(self, layers=[64,128,256,512]):
        super(ReUNet, self).__init__()
        self.block0 = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True)
        )
        self.block1 = Block(64, 128, 2)
        self.block2 = Block(128, 256, 2)
        self.block3 = Block(256, 512, 2)
        self.deconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.block4 = Block(512, 256)
        self.deconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.block5 = Block(256, 128)
        self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.block6 = Block(128, 64)

        self.last = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1,padding=0)
        self.sofmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.block0(x)
        sc1 = x.clone()
        x = self.block1(x)
        sc2 = x.clone()
        x = self.block2(x)
        sc3 = x.clone()
        x = self.block3(x)
        x = self.deconv1(x)
        x = torch.cat((sc3,x), dim=1)
        x = self.block4(x)
        x = self.deconv2(x)
        x = torch.cat((sc2,x), dim=1)
        x = self.block5(x)
        x = self.deconv3(x)
        x = torch.cat((sc1,x), dim=1)
        x = self.block6(x)
        # return self.sofmax(self.last(x))
        return self.last(x) # With Binary Cross Entropy Loss with Logits no need to apply softmax function 
    
# if __name__ == '__main__':
#     x = torch.rand((1,3,224,224))
#     model = ReUNet()
#     ans = model(x)
#     print(ans.shape)
#     print(ans)