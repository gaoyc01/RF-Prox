import torch
import torch.nn as nn
import numpy as np
from torch import nn
from tqdm import *
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from get_data import get_data
from get_data import get_data_fine
from torch.optim.lr_scheduler import StepLR

from model121 import Net121
from modelmlp import Netmlp
from model122 import Net122
from modelcnn import Netcnn
from modeltrans import Nettrans
from model111 import Net111

torch.manual_seed(1)  
model_dir = './model'

array_num = 64
traindata,trainlabel = get_data_fine('./dataset/finetune/test122_fine.mat',array_num)

epoch = 10000
batch_size = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RFDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'CSI': self.data[idx], 'label': self.labels[idx]}
        return sample

scale_factor = 0.015
print('traindata size:{}'.format(traindata.shape))
print('trainlabel size:{}'.format(trainlabel.shape))
trainlabel = (np.tanh(-np.log(scale_factor * trainlabel))+1)/2
trainlabel[trainlabel == 0] = 0.000001
print(trainlabel.shape)
traindataset = RFDataset(traindata,trainlabel.squeeze(0))
train_data_loader = DataLoader(traindataset, batch_size=batch_size, shuffle=True,drop_last=True)
print('Training set processed')

########################################################################
net = torch.load(model_dir+'/base122.pt',map_location = device)
for _,param in enumerate(net.named_parameters()):
    if "cn1" in param[0]:
        param[1].requires_grad = False
    if "cn2" in param[0]:
        param[1].requires_grad = False
    if "inc1" in param[0]:
        param[1].requires_grad = False
    if "mp1" in param[0]:
        param[1].requires_grad = False
    if "mp2" in param[0]:
        param[1].requires_grad = False
    if "mp3" in param[0]:
        param[1].requires_grad = False
    if "ap1" in param[0]:
        param[1].requires_grad = False
    if "c2r" in param[0]:
        param[1].requires_grad = False
    if "encoder_layer" in param[0]:
        param[1].requires_grad = False
    if "trans_encoder" in param[0]:
        param[1].requires_grad = False
    # if "fc1" in param[0]:
    #     param[1].requires_grad = False
print('Model parameters fixed.')

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=1e-4)
scheduler = StepLR(optimizer, step_size=15, gamma=0.5)
###############################################################################################
with torch.no_grad():
    input_torch1 = torch.ones([batch_size*10,1,56,array_num],dtype=torch.complex64).to(device)
    input_torch2 = torch.ones([batch_size*10,1,56,array_num],dtype=torch.complex64).to(device)
    output = net(input_torch1,input_torch2)
print('output size:{}'.format(output.shape))
##############################################################################################

lossf = nn.MSELoss()

def train(net, trainloader, optimizer, criterion, device, iters_all, epoch):
    net.train()
    iter = 0
    for data in tqdm(trainloader,desc=f'Epoch {epoch+1}'):
        inputs, labels = data['CSI'].to(device).to(torch.cfloat), data['label'].to(device).reshape(1,-1)[0].to(torch.float)
        optimizer.zero_grad()
        outputs = net(inputs[:,:10,:,:].unsqueeze(2).reshape(-1,1,56,array_num),inputs[:,10:,:,:].unsqueeze(2).reshape(-1,1,56,array_num)).squeeze(1)
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

writer = SummaryWriter(log_dir="./summary/dont_care")
num_epochs = 60
iters_all = 0
for epoch in range(num_epochs):
    iter_temp = train(net, train_data_loader,optimizer, lossf, device, iters_all, epoch)
    iters_all = iter_temp
    scheduler.step()
    if (epoch+1)%60 == 0:
        torch.save(net,'./model/base122_fine_1000.pt')
    
writer.close()