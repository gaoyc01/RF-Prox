from modelcnn import Netcnn
from modeltrans import Nettrans
from model_smallsize import Netsmall
from model_smallsize22 import Netsmall22
from model_smallsize23 import Netsmall23
from model_smallsize_mlpcat import Netsmallmlp
from thop import profile
import torch
import tqdm
import numpy as np

if __name__ == "__main__":
    device = "cuda:7"
    model = Netsmall().to(device)
    batch_size = 1
    input_torch1 = torch.ones([batch_size*11,1,56,9],dtype=torch.complex64).to(device)
    input_torch2 = torch.ones([batch_size*11,1,56,9],dtype=torch.complex64).to(device)
    flops,params = profile(model,inputs=(input_torch1,input_torch2))
    print('the flops is {}M,the params is {}K'.format(round(flops/(10**6),3), round(params/(10**3),3)))
    
    repetitions = 300
    print('warm up ...\n')
    with torch.no_grad():
        for _ in range(100):
            _ = model(input_torch1,input_torch2)
    torch.cuda.synchronize()        
    
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((repetitions, 1))

    print('testing ...\n')
    with torch.no_grad():
        for rep in tqdm.tqdm(range(repetitions)):
            starter.record()
            _ = model(input_torch1,input_torch2)
            ender.record()
            torch.cuda.synchronize() 
            curr_time = starter.elapsed_time(ender) 
            timings[rep] = curr_time

    avg = timings.sum()/repetitions
    print('\navg={}\n'.format(avg))    
    
    # Netcnn     :the flops is 47.827M,the params is 81.577K
    # Nettrans   :the flops is 5.942M,the params is 267.929K
    # Netsmall   :the flops is 52.648M,the params is 299.305K
    # Netsmall22 :the flops is 31.666M,the params is 297.257K
    # Netsmall23 :the flops is 33.879M,the params is 299.305K
    # Netsmallmlp:the flops is 52.673M,the params is 324.01K
    