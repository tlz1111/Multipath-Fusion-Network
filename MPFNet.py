#MPFNet by TLZ 20250623
import torch as tr
from torch import nn
import torch.nn.functional as F
class_num=8
class Multi_scale(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(Multi_scale, self).__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=1)
        self.conv2=nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(in_channels,out_channels,kernel_size=5,padding=2)
        self.conv=nn.Conv2d(out_channels*3,out_channels,kernel_size=3,padding=1)

    def forward(self,input):
        out_1=self.conv1(input)
        out_2=self.conv2(input)
        out_3=self.conv3(input)
        out=tr.cat([out_1,out_2,out_3],dim=1)
        out=F.relu(out)
        out=self.conv(out)
        return out

class InvertedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride,group):
        super(InvertedBlock, self).__init__()
        channels = expansion * in_channels
        self.stride = stride
        self.basic_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            Multi_scale(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.basic_block(x)
        if self.stride == 1:
            out = out
        return out

class InvertedResidualsBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride,group):
        super(InvertedResidualsBlock, self).__init__()
        channels = expansion * in_channels
        self.stride = stride
        self.basic_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.basic_block(x)
        if self.stride == 1:
            out = out+self.shortcut(x)
        return out

class Mulpath_fusion(nn.Module):
    def __init__(self, input_channels=3):
        super(Mulpath_fusion, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=5, padding=5//2)
        self.bn1=nn.BatchNorm2d(16)
        self.cnn2_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, dilation=5, padding=5)
        self.bn2_1 = nn.BatchNorm2d(16)
        self.cnn2_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, dilation=5, padding=10)
        self.bn2_2 = nn.BatchNorm2d(16)
        self.cnn2_3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, dilation=5, padding=15)
        self.bn2_3 = nn.BatchNorm2d(16)
        self.cnn3=nn.Conv2d(in_channels=32*3, out_channels=128, kernel_size=3, padding=3 // 2)
        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, input):
        out = self.cnn1(input)
        out = F.relu(out,inplace=True)
        out = F.max_pool2d(out, kernel_size=2)
        out_1 = self.cnn2_1(out)
        out_1 = F.relu(out_1,inplace=True)
        out_1 = F.max_pool2d(out_1,kernel_size=2)
        out_2 = self.cnn2_2(out)
        out_2 = F.relu(out_2,inplace=True)
        out_2 = F.max_pool2d(out_2, kernel_size=2)
        out_3 = self.cnn2_3(out)
        out_3 = F.relu(out_3,inplace=True)
        out_3 = F.max_pool2d(out_3, kernel_size=2)
        out = tr.cat([out_1,out_2,out_3], dim=1)
        out=self.cnn3(out)
        out = F.relu(out,inplace=True)
        out = F.max_pool2d(out, kernel_size=4)
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fusion=Mulpath_fusion()
        self.cnn1=nn.Conv2d(in_channels=3,out_channels=32,padding=1,kernel_size=3)
        self.block1=InvertedBlock( in_channels=32, out_channels=64, expansion=2, stride=1,group=4)
        self.block2 = InvertedBlock( in_channels=64, out_channels=128, expansion=2, stride=1,group=4)
        self.block3 = InvertedBlock( in_channels=128, out_channels=256, expansion=2, stride=1,group=8)
        self.block4 = InvertedBlock( in_channels=256, out_channels=512, expansion=2, stride=1,group=16)
        self.block5 = InvertedResidualsBlock( in_channels=512, out_channels=512, expansion=1, stride=1,group=32)
        self.block6 = InvertedResidualsBlock( in_channels=512, out_channels=512, expansion=1, stride=1,group=32)
        self.fc1 = nn.Linear(25088, 1024)
        self.ac=nn.ReLU(inplace=True)
        self.drop=nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 256)
        self.drop2=nn.Dropout(0.5)
        self.fc3=nn.Linear(256,class_num)

    def forward(self, input,batch_size):
      out_1=self.fusion(input)
      out_c=self.cnn1(input)
      out = F.relu(out_c,inplace=True)
      out=self.block1(out)
      out = F.max_pool2d(out, kernel_size=2)
      out = self.block2(out)
      out = F.max_pool2d(out, kernel_size=2)
      out = self.block3(out)
      out = F.max_pool2d(out, kernel_size=2)
      out = self.block4(out)
      out = F.max_pool2d(out, kernel_size=2)
      out = self.block5(out)
      out = F.max_pool2d(out, kernel_size=2)
      out = self.block6(out)
      out_1 = (F.adaptive_avg_pool3d(out_1, output_size=(out.shape[1],out.shape[2],out.shape[3])))
      out = tr.reshape(out, [batch_size, -1])
      out_1 = tr.reshape(out_1, [batch_size, -1])
      out=F.sigmoid(out_1)*out

      out = self.fc1(out)
      out = F.relu(out, inplace=True)
      out=self.drop(out)
      out = self.fc2(out)
      out=F.relu(out,inplace=True)
      out=self.drop2(out)
      out=self.fc3(out)
      out=F.softmax(out,dim=1)
      return out

