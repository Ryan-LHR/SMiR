'''
model structure
'''
import torch
import torch.nn as nn

# region models for fm
class fm_simple_Net(torch.nn.Module):
    '''
    The original model structure of simple_fm
    From Arachne
    '''
    def __init__(self):
        super(fm_simple_Net, self).__init__()
        self.fc_v1 = nn.Linear(784, 100)
        self.relu = nn.ReLU()
        self.fc_v2 = nn.Linear(100, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.fc_v1(x)
        x = self.relu(x)
        x = self.fc_v2(x)
        return x

class fm_simple_Net_withForword2(torch.nn.Module):
    '''
    The original model structure of simple_fm, added fc layers
    From Arachne
    '''
    def __init__(self):
        super(fm_simple_Net_withForword2, self).__init__()
        self.fc_v1 = nn.Linear(784, 100)
        self.relu = nn.ReLU()
        self.fc_v2 = nn.Linear(100, 10)
        # added fc layer
        self.fc_v3 = nn.Linear(100, 2)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.fc_v1(x)
        x = self.relu(x)
        x = self.fc_v2(x)
        return x

    def forward_v2(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.fc_v1(x)
        x = self.relu(x)
        x = self.fc_v3(x)
        return x

class fm_simple_Net_feature_to_n(torch.nn.Module):
    '''
    The original model structure of fm_simple
    From Arachne
    '''
    def __init__(self):
        super().__init__()
        self.fc_v2 = nn.Linear(100, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.fc_v2(x)
        return x

class fm_simple_Net_feature_to_2(torch.nn.Module):
    '''
    The original model structure of fm_simple
    From Arachne
    '''
    def __init__(self):
        super().__init__()
        self.fc_v3 = nn.Linear(100, 2)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.fc_v3(x)
        return x

class fm_simple_Net_feature_to_2_v2(torch.nn.Module):
    '''
    The original model structure of fm_simple
    From Arachne
    '''
    def __init__(self):
        super().__init__()
        self.fc_v3 = nn.Linear(100, 32)
        self.fc_v4 = nn.Linear(32, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.fc_v3(x)
        x = self.relu(x)
        x = self.fc_v4(x)
        return x
# endregion models for fm

# region models for simple_cm
class c10_simple_Net(torch.nn.Module):
    '''
    The original model structure of simple_cm
    From Arachne
    '''
    def __init__(self):
        super(c10_simple_Net, self).__init__()
        # self.zero_pad = nn.ConstantPad2d(padding=(1, 1, 1, 1), value=0)  # 各一列的zero padding
        self.conv_v1 = nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='zeros')
        self.bn = nn.BatchNorm2d(num_features=16)
        self.relu = nn.ReLU()
        self.fc_v1 = nn.Linear(1024, 512)
        self.fc_v2 = nn.Linear(512, 10)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4), ceil_mode=True)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4), ceil_mode=False)

    def forward(self, x):
        batch_size = x.size(0)
        # x = x.view(batch_size, -1)
        x = self.conv_v1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        # x = x.reshape(1024)  # reshape
        x = x.view(batch_size, -1)
        x = self.fc_v1(x)
        x = self.relu(x)
        x = self.fc_v2(x)
        return x

class c10_simple_Net_withForword2(torch.nn.Module):
    '''
    The original model structure of simple_cm, added fc layers
    From Arachne
    '''
    def __init__(self):
        super(c10_simple_Net_withForword2, self).__init__()
        # self.zero_pad = nn.ConstantPad2d(padding=(1, 1, 1, 1), value=0)  # 各一列的zero padding
        self.conv_v1 = nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='zeros')
        self.bn = nn.BatchNorm2d(num_features=16)
        self.relu = nn.ReLU()
        self.fc_v1 = nn.Linear(1024, 512)
        self.fc_v2 = nn.Linear(512, 10)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4), ceil_mode=True)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4), ceil_mode=False)
        # added fc layers
        self.fc_v3 = nn.Linear(1024, 512)
        self.fc_v4 = nn.Linear(512, 2)

    def forward(self, x):
        batch_size = x.size(0)
        # x = x.view(batch_size, -1)
        x = self.conv_v1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        # x = x.reshape(1024)  # reshape
        x = x.view(batch_size, -1)
        x = self.fc_v1(x)
        x = self.relu(x)
        x = self.fc_v2(x)
        return x

    def forward_v2(self, x):
        batch_size = x.size(0)
        # x = x.view(batch_size, -1)
        x = self.conv_v1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        # x = x.reshape(1024)  # reshape
        x = x.view(batch_size, -1)
        x = self.fc_v3(x)
        x = self.relu(x)
        x = self.fc_v4(x)
        return x

    def forward_v3(self, x):
        batch_size = x.size(0)
        # x = x.view(batch_size, -1)
        x = self.conv_v1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        # x = x.reshape(1024)  # reshape
        x = x.view(batch_size, -1)
        x = self.fc_v1(x)
        x = self.relu(x)
        x = self.fc_v4(x)
        return x

class c10_simple_Net_withForword2_v2(torch.nn.Module):
    '''
    The original model structure of simple_cm, added fc layers
    From Arachne
    '''
    def __init__(self):
        super().__init__()
        self.conv_v1 = nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='zeros')
        self.bn = nn.BatchNorm2d(num_features=16)
        self.relu_v1 = nn.ReLU()
        self.relu_v2 = nn.ReLU()
        self.fc_v1 = nn.Linear(1024, 512)
        self.fc_v2 = nn.Linear(512, 10)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4), ceil_mode=True)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4), ceil_mode=False)
        # added fc layers
        self.fc_v3 = nn.Linear(1024, 512)
        self.fc_v4 = nn.Linear(512, 2)
        self.relu_v3 = nn.ReLU()
        self.relu_v4 = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        # x = x.view(batch_size, -1)
        x = self.conv_v1(x)
        x = self.bn(x)
        x = self.relu_v1(x)
        x = self.maxpool1(x)
        # x = x.reshape(1024)  # reshape
        x = x.view(batch_size, -1)
        x = self.fc_v1(x)
        x = self.relu_v2(x)
        x = self.fc_v2(x)
        return x

    def forward_v2(self, x):
        batch_size = x.size(0)
        # x = x.view(batch_size, -1)
        x = self.conv_v1(x)
        x = self.bn(x)
        x = self.relu_v3(x)
        x = self.maxpool1(x)
        # x = x.reshape(1024)  # reshape
        x = x.view(batch_size, -1)
        x = self.fc_v3(x)
        x = self.relu_v4(x)
        x = self.fc_v4(x)
        return x

class c10_simple_Net_withForword2_v3(torch.nn.Module):
    '''
    The original model structure of simple_cm, added fc layers
    From Arachne
    '''
    def __init__(self):
        super().__init__()
        self.relu_v1 = nn.ReLU()
        self.relu_v2 = nn.ReLU()
        self.fc_v1 = nn.Linear(1024, 512)
        self.fc_v2 = nn.Linear(512, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.fc_v1(x)
        x = self.relu_v2(x)
        x = self.fc_v2(x)
        return x

class c10_simple_Net_feature_to_2(torch.nn.Module):
    '''
    The original model structure of simple_cm, added fc layers
    From Arachne
    '''
    def __init__(self):
        super().__init__()
        # added fc layers
        self.fc_v3 = nn.Linear(1024, 512)
        self.fc_v4 = nn.Linear(512, 2)
        self.relu_v3 = nn.ReLU()
        self.relu_v4 = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.fc_v3(x)
        x = self.relu_v3(x)
        x = self.fc_v4(x)
        return x
# endregion models for simple_cm

# region models for C10_CNN1
class C10_CNN1_Net(torch.nn.Module):
    '''
    The original model structure of CNN1
    From Arachne and Apricot
    '''
    def __init__(self):
        super(C10_CNN1_Net, self).__init__()
        self.conv_v1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='zeros')
        self.conv_v2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='zeros')
        self.conv_v3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='zeros')
        self.conv_v4 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='zeros')
        self.maxpool_v1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), ceil_mode=True)
        self.fc_v1 = nn.Linear(8192, 256)
        self.fc_v2 = nn.Linear(256, 256)
        self.fc_v3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv_v1(x)
        x = self.relu(x)
        x = self.conv_v2(x)
        x = self.relu(x)
        x = self.maxpool_v1(x)
        x = self.conv_v3(x)
        x = self.relu(x)
        x = self.conv_v4(x)
        x = self.relu(x)
        x = self.maxpool_v1(x)

        x = x.view(batch_size, -1)
        x = self.fc_v1(x)
        x = self.relu(x)
        x = self.fc_v2(x)
        x = self.relu(x)
        x = self.fc_v3(x)
        return x

class C10_CNN1_Net_withForword2(torch.nn.Module):
    '''
    The original model structure of CNN1, added fc layers
    From Arachne and Apricot
    '''
    def __init__(self):
        super(C10_CNN1_Net_withForword2, self).__init__()
        self.conv_v1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='zeros')
        self.conv_v2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='zeros')
        self.conv_v3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='zeros')
        self.conv_v4 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='zeros')
        self.maxpool_v1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), ceil_mode=True)
        self.fc_v1 = nn.Linear(8192, 256)
        self.fc_v2 = nn.Linear(256, 256)
        self.fc_v3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        # added fc layers
        self.fc_v4 = nn.Linear(8192, 256)
        self.fc_v5 = nn.Linear(256, 256)
        self.fc_v6 = nn.Linear(256, 2)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv_v1(x)
        x = self.relu(x)
        x = self.conv_v2(x)
        x = self.relu(x)
        x = self.maxpool_v1(x)
        x = self.conv_v3(x)
        x = self.relu(x)
        x = self.conv_v4(x)
        x = self.relu(x)
        x = self.maxpool_v1(x)

        x = x.view(batch_size, -1)
        x = self.fc_v1(x)
        x = self.relu(x)
        x = self.fc_v2(x)
        x = self.relu(x)
        x = self.fc_v3(x)
        return x

    def forward_v2(self, x):
        batch_size = x.size(0)
        x = self.conv_v1(x)
        x = self.relu(x)
        x = self.conv_v2(x)
        x = self.relu(x)
        x = self.maxpool_v1(x)
        x = self.conv_v3(x)
        x = self.relu(x)
        x = self.conv_v4(x)
        x = self.relu(x)
        x = self.maxpool_v1(x)

        x = x.view(batch_size, -1)
        x = self.fc_v4(x)
        x = self.relu(x)
        x = self.fc_v5(x)
        x = self.relu(x)
        x = self.fc_v6(x)
        return x

class C10_CNN1_Net_withForword2_v2(torch.nn.Module):
    '''
    The original model structure of CNN1, added fc layers
    From Arachne and Apricot
    '''
    def __init__(self):
        super(C10_CNN1_Net_withForword2_v2, self).__init__()
        self.conv_v1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='zeros')
        self.conv_v2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='zeros')
        self.conv_v3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='zeros')
        self.conv_v4 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='zeros')
        self.maxpool_v1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), ceil_mode=True)
        self.maxpool_v2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), ceil_mode=True)
        self.fc_v1 = nn.Linear(8192, 256)
        self.fc_v2 = nn.Linear(256, 256)
        self.fc_v3 = nn.Linear(256, 10)
        self.relu_v1 = nn.ReLU()
        self.relu_v2 = nn.ReLU()
        self.relu_v3 = nn.ReLU()
        self.relu_v4 = nn.ReLU()
        self.relu_v5 = nn.ReLU()
        self.relu_v6 = nn.ReLU()

        # added fc layers
        self.fc_v4 = nn.Linear(8192, 256)
        self.fc_v5 = nn.Linear(256, 256)
        self.fc_v6 = nn.Linear(256, 2)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv_v1(x)
        x = self.relu_v1(x)
        x = self.conv_v2(x)
        x = self.relu_v2(x)
        x = self.maxpool_v1(x)
        x = self.conv_v3(x)
        x = self.relu_v3(x)
        x = self.conv_v4(x)
        x = self.relu_v4(x)
        x = self.maxpool_v2(x)

        x = x.view(batch_size, -1)
        x = self.fc_v1(x)
        x = self.relu_v5(x)
        x = self.fc_v2(x)
        x = self.relu_v6(x)
        x = self.fc_v3(x)
        return x

    def forward_v2(self, x):
        batch_size = x.size(0)
        x = self.conv_v1(x)
        x = self.relu_v1(x)
        x = self.conv_v2(x)
        x = self.relu_v1(x)
        x = self.maxpool_v1(x)
        x = self.conv_v3(x)
        x = self.relu_v1(x)
        x = self.conv_v4(x)
        x = self.relu_v1(x)
        x = self.maxpool_v1(x)

        x = x.view(batch_size, -1)
        x = self.fc_v4(x)
        x = self.relu_v1(x)
        x = self.fc_v5(x)
        x = self.relu_v1(x)
        x = self.fc_v6(x)
        return x

class C10_CNN1_Net_feature_to_n(torch.nn.Module):
    '''
    The original model structure of C10_CNN1, added fc layers
    From Arachne
    '''
    def __init__(self):
        super().__init__()
        self.fc_v1 = nn.Linear(8192, 256)
        self.fc_v2 = nn.Linear(256, 256)
        self.fc_v3 = nn.Linear(256, 10)
        self.relu_v5 = nn.ReLU()
        self.relu_v6 = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.fc_v1(x)
        x = self.relu_v5(x)
        x = self.fc_v2(x)
        x = self.relu_v6(x)
        x = self.fc_v3(x)
        return x

class C10_CNN1_Net_feature_to_2(torch.nn.Module):
    '''
    The original model structure of C10_CNN1, added fc layers
    From Arachne
    '''
    def __init__(self):
        super().__init__()

        self.fc_v4 = nn.Linear(8192, 256)
        self.fc_v5 = nn.Linear(256, 256)
        self.fc_v6 = nn.Linear(256, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.fc_v4(x)
        x = self.relu(x)
        x = self.fc_v5(x)
        x = self.relu(x)
        x = self.fc_v6(x)
        return x

# endregion models for C10_CNN1

# region models for C10_CNN2 & C10_CNN3
class C10_CNN2_Net(torch.nn.Module):
    '''
    The original model structure of CNN2
    From Arachne and Apricot
    '''
    def __init__(self):
        super(C10_CNN2_Net, self).__init__()
        self.conv_v1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='zeros')
        self.conv_v2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='zeros')
        self.conv_v3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='zeros')
        self.conv_v4 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='zeros')
        self.conv_v5 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='zeros')

        self.bn_v1 = nn.BatchNorm2d(num_features=32)
        self.bn_v2 = nn.BatchNorm2d(num_features=32)
        self.bn_v3 = nn.BatchNorm2d(num_features=64)
        self.bn_v4 = nn.BatchNorm2d(num_features=128)
        self.bn_v5 = nn.BatchNorm2d(num_features=128)

        self.maxpool_v1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), ceil_mode=True)
        self.fc_v1 = nn.Linear(2048, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv_v1(x)
        x = self.bn_v1(x)
        x = self.relu(x)
        x = self.conv_v2(x)
        x = self.bn_v2(x)
        x = self.relu(x)
        x = self.maxpool_v1(x)

        x = self.conv_v3(x)
        x = self.bn_v3(x)
        x = self.relu(x)
        x = self.maxpool_v1(x)

        x = self.conv_v4(x)
        x = self.bn_v4(x)
        x = self.relu(x)
        x = self.conv_v5(x)
        x = self.bn_v5(x)
        x = self.relu(x)
        x = self.maxpool_v1(x)

        x = x.view(batch_size, -1)
        x = self.fc_v1(x)
        return x

class C10_CNN3_Net(torch.nn.Module):
    '''
    The original model structure of CNN13
    From Arachne and Apricot
    '''
    def __init__(self):
        super(C10_CNN3_Net, self).__init__()
        self.conv_v1 = nn.Conv2d(3, 96, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='zeros')
        self.conv_v2 = nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='zeros')
        self.conv_v3 = nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='zeros')
        self.conv_v4 = nn.Conv2d(96, 192, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='zeros')
        self.conv_v5 = nn.Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='zeros')
        self.conv_v6 = nn.Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='zeros')
        self.conv_v7 = nn.Conv2d(192, 10, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='zeros')

        self.maxpool_v1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), ceil_mode=True)
        self.fc_v1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv_v1(x)
        x = self.relu(x)
        x = self.conv_v2(x)
        x = self.relu(x)
        x = self.conv_v3(x)
        x = self.relu(x)
        x = self.maxpool_v1(x)

        x = self.conv_v4(x)
        x = self.relu(x)
        x = self.conv_v5(x)
        x = self.relu(x)
        x = self.conv_v6(x)
        x = self.relu(x)
        x = self.maxpool_v1(x)

        x = self.conv_v7(x)
        x = self.relu(x)

        x = x.view(batch_size, -1)
        x = self.fc_v1(x)
        return x
# endregion models for C10_CNN2 & C10_CNN3

# region models for simple_gtsrb
class gtsrb_simple_Net_withForword2(torch.nn.Module):
    '''
    The original model structure of simple_gtsrb, added fc layers
    From Arachne
    '''
    def __init__(self):
        super().__init__()
        self.conv_v1 = nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1))
        self.relu_v1 = nn.ReLU()
        self.bn_v1 = nn.BatchNorm2d(num_features=16)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), ceil_mode=True)

        self.fc_v1 = nn.Linear(8464, 512)
        self.relu_v2 = nn.ReLU()
        self.bn_v2 = nn.BatchNorm1d(num_features=512)
        self.fc_v2 = nn.Linear(512, 43)
        self.softmax_v1 = nn.Softmax()

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv_v1(x)
        x = self.relu_v1(x)
        x = self.bn_v1(x)
        x = self.maxpool1(x)

        x = x.view(batch_size, -1)
        x = self.fc_v1(x)
        x = self.relu_v2(x)
        x = self.bn_v2(x)
        x = self.fc_v2(x)
        x = self.softmax_v1(x)
        return x

class gtsrb_simple_Net_feature_to_n(torch.nn.Module):
    '''
    The original model structure of simple_gtsrb, added fc layers
    From Arachne
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return x

class gtsrb_simple_Net_feature_to_2(torch.nn.Module):
    '''
    The original model structure of simple_gtsrb, added fc layers
    From Arachne
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return x
# endregion models for simple_gtsrb

# region models for aptos_CNN

class aptos_resnet18_Net_new(nn.Module):
    def __init__(self, num_classes=5):
        super(aptos_resnet18_Net_new, self).__init__()

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layer 1 (no downsampling)
        # Block 1
        self.conv2_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.relu2_2 = nn.ReLU(inplace=True)

        # Block 2
        self.conv2_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_3 = nn.BatchNorm2d(64)
        self.relu2_3 = nn.ReLU(inplace=True)
        self.conv2_4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_4 = nn.BatchNorm2d(64)
        self.relu2_4 = nn.ReLU(inplace=True)

        # Layer 2 (with downsampling)
        # Block 1
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.shortcut3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128)
        )

        # Block 2
        self.conv3_3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_3 = nn.BatchNorm2d(128)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.conv3_4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_4 = nn.BatchNorm2d(128)
        self.relu3_4 = nn.ReLU(inplace=True)

        # Layer 3 (with downsampling)
        # Block 1
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.shortcut4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256)
        )

        # Block 2
        self.conv4_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_3 = nn.BatchNorm2d(256)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.conv4_4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_4 = nn.BatchNorm2d(256)
        self.relu4_4 = nn.ReLU(inplace=True)

        # Layer 4 (with downsampling)
        # Block 1
        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.shortcut5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512)
        )

        # Block 2
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5_3 = nn.BatchNorm2d(512)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5_4 = nn.BatchNorm2d(512)
        self.relu5_4 = nn.ReLU(inplace=True)

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.dropout_v1 = nn.Dropout(0.5)

    def forward(self, x):
        # Initial layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        # Layer 1
        out = self.relu2_1(self.bn2_1(self.conv2_1(x)))
        out = self.bn2_2(self.conv2_2(out))
        x = self.relu2_2(out + x)

        out = self.relu2_3(self.bn2_3(self.conv2_3(x)))
        out = self.bn2_4(self.conv2_4(out))
        x = self.relu2_4(out + x)

        # Layer 2
        out = self.relu3_1(self.bn3_1(self.conv3_1(x)))
        out = self.bn3_2(self.conv3_2(out))
        x = self.relu3_2(out + self.shortcut3(x))

        out = self.relu3_3(self.bn3_3(self.conv3_3(x)))
        out = self.bn3_4(self.conv3_4(out))
        x = self.relu3_4(out + x)

        # Layer 3
        out = self.relu4_1(self.bn4_1(self.conv4_1(x)))
        out = self.bn4_2(self.conv4_2(out))
        x = self.relu4_2(out + self.shortcut4(x))

        out = self.relu4_3(self.bn4_3(self.conv4_3(x)))
        out = self.bn4_4(self.conv4_4(out))
        x = self.relu4_4(out + x)

        # Layer 4
        out = self.relu5_1(self.bn5_1(self.conv5_1(x)))
        out = self.bn5_2(self.conv5_2(out))
        x = self.relu5_2(out + self.shortcut5(x))

        out = self.relu5_3(self.bn5_3(self.conv5_3(x)))
        out = self.bn5_4(self.conv5_4(out))
        x = self.relu5_4(out + x)

        # Final layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.dropout_v1(x)
        x = self.fc(x)

        return x

class aptos_resnet18_Net_new_feature_to_2(nn.Module):
    def __init__(self, num_classes=5):
        super(aptos_resnet18_Net_new_feature_to_2, self).__init__()

        self.fc = nn.Linear(512, 2)
        self.dropout_v1 = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.flatten(x, 1)
        # x = self.dropout_v1(x)
        x = self.fc(x)

        return x
# endregion models for aptos_CNN

# region models for neu_cls_CNN
class neucls_Net_withForword2(torch.nn.Module):
    '''
    model for neucls(200*200), added fc layers
    '''
    def __init__(self):
        super().__init__()
        self.conv_v1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='zeros')
        self.conv_v1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='zeros')
        self.relu_v1 = nn.ReLU()
        self.bn_v1 = nn.BatchNorm2d(num_features=16)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), ceil_mode=True)

        self.fc_v1 = nn.Linear(8464, 512)
        self.relu_v2 = nn.ReLU()
        self.bn_v2 = nn.BatchNorm1d(num_features=512)
        self.fc_v2 = nn.Linear(512, 43)
        self.softmax_v1 = nn.Softmax()

        # added fc layers
        self.fc_v3 = nn.Linear(8464, 512)
        self.fc_v4 = nn.Linear(512, 2)
        self.relu_v4 = nn.ReLU()
        self.softmax_v2 = nn.Softmax()
        self.bn_v3 = nn.BatchNorm1d(num_features=512)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.softmax_v1(x)
        return x

    def forward_v2(self, x):
        batch_size = x.size(0)

        return x


class neucls64_Net_withForword2_new(torch.nn.Module):
    '''
    Simplified model for neucls(64*64), reduced to 2 conv layers and added fc layers
    '''

    def __init__(self):
        super().__init__()

        self.conv_v1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv_v2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        self.maxpool_v1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.maxpool_v2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc_v1 = nn.Linear(32 * 16 * 16, 256)  # Adjusted based on the new feature map size
        self.fc_v2 = nn.Linear(256, 6)

        self.relu_v1 = nn.ReLU()
        self.relu_v2 = nn.ReLU()
        self.relu_v3 = nn.ReLU()
        self.dropout_v1 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.maxpool_v1(self.relu_v1(self.conv_v1(x)))
        x = self.maxpool_v2(self.relu_v2(self.conv_v2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu_v3(self.fc_v1(x))
        x = self.fc_v2(x)

        return x

class neucls64_Net_feature_to_2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_v1 = nn.Linear(32 * 16 * 16, 256)
        self.fc_v2 = nn.Linear(256, 2)
        self.relu_v3 = nn.ReLU()
        self.dropout_v1 = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu_v3(self.fc_v1(x))
        x = self.fc_v2(x)
        return x

class template(torch.nn.Module):
    '''
    The model structure of a template model
    '''
    def __init__(self):
        super(template, self).__init__()

    def forward(self, x):
        return x

class MLP_Model(nn.Module):
    def __init__(self, input_dimensions):
        super(MLP_Model, self).__init__()
        self.fc1 = nn.Linear(input_dimensions, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, input_dimensions)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DirectOutputModel(nn.Module):
    def __init__(self):
        super(DirectOutputModel, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return x