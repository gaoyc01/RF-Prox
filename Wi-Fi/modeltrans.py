import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from complex import complex_layers as cl 
from complex import complex_functions as cf 

def init_weight_norm(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def init_weight_zero(module):
    if isinstance(module, nn.Linear):
        nn.init.constant_(module.weight, 0)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def init_weight_xavier(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
            
## Model
class ComplexResidual2d(nn.Module):
    
    def __init__(self, input_channels, num_channels, kernel_size, padding, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=kernel_size, padding=padding, stride=stride, dtype=torch.cfloat)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=kernel_size, padding=padding, stride=stride, dtype=torch.cfloat)
        self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=stride, dtype=torch.cfloat)
        self.bn1 = cl.NaiveComplexBatchNorm2d(num_channels)
        self.bn2 = cl.NaiveComplexBatchNorm2d(num_channels)
        self.apply(init_weight_xavier)
    def forward(self, X):
        Y = cf.complex_tanh(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y)) + self.conv3(X)
        return cf.complex_tanh(Y)
class ComplexInception2d(nn.Module):

    def __init__(self, input_channels, p1, p2, p3, p4, **kwargs):
        super().__init__(**kwargs)
        self.p1_1 = ComplexResidual2d(input_channels, p1["1-channel"], p1["1-kernel"][0], p1["1-kernel"][1])
        self.p2_1 = ComplexResidual2d(input_channels, p2["1-channel"], p2["1-kernel"][0], p2["1-kernel"][1])
        self.p2_2 = ComplexResidual2d(p2["1-channel"], p2["2-channel"], p2["2-kernel"][0], p2["2-kernel"][1])
        self.p3_1 = ComplexResidual2d(input_channels, p3["1-channel"], p3["1-kernel"][0], p3["1-kernel"][1])
        self.p3_2 = ComplexResidual2d(p3["1-channel"], p3["2-channel"], p3["2-kernel"][0], p3["2-kernel"][1])
        self.p4_1 = cl.ComplexMaxPool2d(kernel_size=p4["1-kernel"][0], padding=p4["1-kernel"][1], stride=1)
        self.p4_2 = ComplexResidual2d(input_channels, p4["2-channel"], p4["2-kernel"][0], p4["2-kernel"][1])
        self.apply(init_weight_xavier)
    def forward(self, X):
        p1 = cf.complex_tanh(self.p1_1(X))
        p2 = cf.complex_tanh(self.p2_2(cf.complex_tanh(self.p2_1(X))))
        p3_1 = self.p3_1(X)
        p3_1_relu = cf.complex_tanh(p3_1)
        p3_2 = self.p3_2(p3_1_relu)
        p3 = cf.complex_tanh(p3_2)
        p4 = cf.complex_tanh(self.p4_2(cf.complex_tanh(self.p4_1(X))))
        return torch.cat((p1, p2, p3, p4), dim=1)   
class ComplexCNN2d(nn.Module):

    def __init__(self, input_channels, num_channels, config):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, num_channels, kernel_size=config["1-kernel"][0], padding=config["1-kernel"][1], stride=config["1-stride"], dtype=torch.cfloat)
        self.bn = cl.NaiveComplexBatchNorm2d(num_channels)
        self.apply(init_weight_xavier)
    def forward(self, X):
        return self.bn(self.conv(X))
class Complex2Real(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 1)
        self.apply(init_weight_zero)
    def forward(self, X):
        X = X.unsqueeze(2)
        X = self.linear1(torch.cat((X.real, X.imag), dim=2))
        X = self.linear2(F.relu(X))
        return X.squeeze(2)
class Nettrans(nn.Module):
    """
    Artificial neural network object. 
    
    Args:
        - multiple NN blocks and other parameters. 
    Return:
        - network model for training, validation and testing.
    """
    def __init__(self):
        super(Nettrans, self).__init__()
        self.fc1 = nn.Linear(56*9, 128, dtype=torch.cfloat)
        self.fc2 = nn.Linear(128, 16)
        self.c2r = Complex2Real() 
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=128)
        self.trans_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.apply(init_weight_xavier)
    def encoder(self, X):# X [batch_size*10,1,64,64]
        X =X.reshape(-1,56*9)
        X = cf.complex_leaky_relu(self.fc1(X),0.05)
        X = self.c2r(X)
        X = X.reshape(-1,11,128).permute(1,0,2) # (sequence length, batch size, feature size)
        X = self.trans_encoder(X).permute(1,0,2)
        X = F.leaky_relu(self.fc2(X),0.05)
        X = X.reshape(-1,176)
        return X

    def forward(self, Y1,Y2):
        out1 = self.encoder(Y1)
        out2 = self.encoder(Y2)
        out = F.cosine_similarity(out1, out2, dim=1).unsqueeze(1)
        return out