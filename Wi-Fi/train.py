import torch
import torch.nn as nn
import numpy as np
from torch import nn
from tqdm import *
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from get_data import get_data
from torch.optim.lr_scheduler import StepLR

from model_smallsize22 import Netsmall22
from model_smallsize_mlpcat import Netsmallmlp
from model_smallsize23 import Netsmall23
from model_smallsize34 import Netsmall34
from modelcnn import Netcnn
from modeltrans import Nettrans
from model_smallsize import Netsmall

torch.manual_seed(1)  
model_dir = './model'

array_num = 9
traindata,trainlabel = get_data('./dataset/train/train33.mat',array_num)

epoch = 10000
batch_size = 64
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

class RFDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'CSI': self.data[idx], 'label': self.labels[idx]}
        return sample

scale_factor = 0.5
print('traindata size:{}'.format(traindata.shape))
print('trainlabel size:{}'.format(trainlabel.shape))
trainlabel = (np.tanh(-np.log(scale_factor * trainlabel))+1)/2
print(trainlabel.shape)
traindataset = RFDataset(traindata,trainlabel.squeeze(1))
train_data_loader = DataLoader(traindataset, batch_size=batch_size, shuffle=True,drop_last=True)
print('Training set processed')

net = Netsmall().to(device)
###############################################################################################
with torch.no_grad():
    input_torch1 = torch.ones([batch_size*11,1,56,array_num],dtype=torch.complex64).to(device)
    input_torch2 = torch.ones([batch_size*11,1,56,array_num],dtype=torch.complex64).to(device)

    output = net(input_torch1,input_torch2)
print('output size:{}'.format(output.shape))
##############################################################################################

optimizer = torch.optim.AdamW(net.parameters(),lr=1e-4)
scheduler = StepLR(optimizer, step_size=15, gamma=0.5)
lossf = nn.MSELoss()

def train(net, trainloader, optimizer, criterion, device, iters_all, epoch):
    net.train()
    iter = 0
    for data in tqdm(trainloader,desc=f'Epoch {epoch+1}'):
        inputs, labels = data['CSI'].to(device).to(torch.cfloat), data['label'].to(device).reshape(1,-1)[0].to(torch.float)
        optimizer.zero_grad()
        outputs = net(inputs[:,:11,:,:].unsqueeze(2).reshape(-1,1,56,array_num),inputs[:,11:,:,:].unsqueeze(2).reshape(-1,1,56,array_num)).squeeze(1)
        if iter%600 == 0:
            print(outputs[:10])
            print(labels[:10])
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        writer.add_scalar("loss", loss.detach(), iters_all + iter)
        iter = iter + 1 
    iters_all = iters_all + iter
    return iters_all

# Training loop

writer = SummaryWriter(log_dir="./summary/base33")
num_epochs = 200
iters_all = 0
for epoch in range(num_epochs):
    iter_temp = train(net, train_data_loader,optimizer, lossf, device, iters_all, epoch)
    iters_all = iter_temp
    scheduler.step()
    if (epoch+1)%100 == 0:
        torch.save(net,'./model/base33_{}.pt'.format(epoch+1))
    
writer.close()
