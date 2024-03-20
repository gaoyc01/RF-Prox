import torch
import numpy as np
import numpy as np
from tqdm import *
from torch.utils.data import Dataset, DataLoader
from get_data import get_data_test
from argparse import ArgumentParser
import os
class RFDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'CSI': self.data[idx], 'label': self.labels[idx]}
        return sample

def rank_elements_min(seq):
    sorted_seq = sorted((value, index) for index, value in enumerate(seq))
    rank_dict = {value[1]: rank + 1 for rank, value in enumerate(sorted_seq)}
    return np.array([rank_dict[index] for index in range(len(seq))])

def rank_elements_max(seq):
    sorted_seq = sorted((value, index) for index, value in enumerate(seq))
    rank_dict = {value[1]: rank + 1 for rank, value in enumerate(sorted_seq)}
    return len(seq)+1-np.array([rank_dict[index] for index in range(len(seq))])

def LIS(outlist,loclist):
    num = len(loclist)
    Coef = [1/np.log2(val+1) for val in loclist]
    result = np.sum((num-abs(np.array(outlist)-np.array(loclist)))/num*Coef/np.sum(Coef))
    return result
def acc(outlist,loclist):
    outlist = np.array(outlist)
    loclist = np.array(loclist)
    if outlist[np.where(loclist==1)[0][0]]==1:
        return 1
    else:
        return 0
def test(args):
    device = 'cuda:2'
    model_dir = args.model_dir
    model_name = args.model_name
    result_dir = args.result_dir
    dataset_dir = args.dataset_dir
    dataset_name = args.dataset_name
    result_name = 'model_'+model_name.split('.')[0]+'_dataset_'+dataset_name.split('.')[0]+'.txt'
    array_num = 16
    
    net = torch.load(model_dir+model_name,map_location = device)
    
    testdata, testlabel = get_data_test(dataset_dir+dataset_name,array_num)
    num = len(testdata)
    test_num = 200 

    testrand = np.random.randint(0, num, (11,test_num))
    testdata = np.concatenate((testdata[testrand[0]],testdata[testrand[1]],\
                                testdata[testrand[2]],testdata[testrand[3]],\
                                testdata[testrand[4]],testdata[testrand[5]],\
                                testdata[testrand[6]],testdata[testrand[7]],\
                                testdata[testrand[8]],testdata[testrand[9]],testdata[testrand[10]]),axis=1) # (2000, 10*6, 128, 64)
    
    testloc = np.concatenate((testlabel[testrand[0]],testlabel[testrand[1]],\
                                testlabel[testrand[2]],testlabel[testrand[3]],\
                                testlabel[testrand[4]],testlabel[testrand[5]],\
                                testlabel[testrand[6]],testlabel[testrand[7]],\
                                testlabel[testrand[8]],testlabel[testrand[9]],testlabel[testrand[10]]),axis=1) # (2000, 3*6)
    testdataset = RFDataset(testdata,testloc)
    testloader = DataLoader(testdataset, batch_size=1, shuffle=False,drop_last=True)
    print('testdata size：{}'.format(testdata.shape))
    print('testloc size：{}'.format(testloc.shape))
    print('Test set processing complete')
    
    net.eval()
    testiter = 0
    nums = len(testloader)
    repeat_times = 10
    accuracy9 = np.zeros([1,repeat_times])
    accuracy8 = np.zeros([1,repeat_times])
    accuracy7 = np.zeros([1,repeat_times])
    accuracy6 = np.zeros([1,repeat_times])
    accuracy5 = np.zeros([1,repeat_times])
    accuracy4 = np.zeros([1,repeat_times])
    accuracy3 = np.zeros([1,repeat_times])
    accuracy2 = np.zeros([1,repeat_times])
    ndcg2 = np.zeros([1,repeat_times])
    ndcg3 = np.zeros([1,repeat_times])
    ndcg4 = np.zeros([1,repeat_times])
    ndcg5 = np.zeros([1,repeat_times])
    ndcg6 = np.zeros([1,repeat_times])
    ndcg7 = np.zeros([1,repeat_times])
    ndcg8 = np.zeros([1,repeat_times])
    ndcg9 = np.zeros([1,repeat_times])
    with torch.no_grad():
        for data in tqdm(testloader,desc=f'Epoch {testiter // len(testloader)}'):
            inputs, labels = data['CSI'].to(device).to(torch.cfloat), data['label'].reshape(1,-1)[0]
            out1 = net(inputs[:,:10,:,:].unsqueeze(2).reshape(-1,1,56,array_num),inputs[:,10:20,:,:].unsqueeze(2).reshape(-1,1,56,array_num))[0][0].item()
            out2 = net(inputs[:,:10,:,:].unsqueeze(2).reshape(-1,1,56,array_num),inputs[:,20:30,:,:].unsqueeze(2).reshape(-1,1,56,array_num))[0][0].item()
            out3 = net(inputs[:,:10,:,:].unsqueeze(2).reshape(-1,1,56,array_num),inputs[:,30:40,:,:].unsqueeze(2).reshape(-1,1,56,array_num))[0][0].item()
            out4 = net(inputs[:,:10,:,:].unsqueeze(2).reshape(-1,1,56,array_num),inputs[:,40:50,:,:].unsqueeze(2).reshape(-1,1,56,array_num))[0][0].item()
            out5 = net(inputs[:,:10,:,:].unsqueeze(2).reshape(-1,1,56,array_num),inputs[:,50:60,:,:].unsqueeze(2).reshape(-1,1,56,array_num))[0][0].item()
            out6 = net(inputs[:,:10,:,:].unsqueeze(2).reshape(-1,1,56,array_num),inputs[:,60:70,:,:].unsqueeze(2).reshape(-1,1,56,array_num))[0][0].item()
            out7 = net(inputs[:,:10,:,:].unsqueeze(2).reshape(-1,1,56,array_num),inputs[:,70:80,:,:].unsqueeze(2).reshape(-1,1,56,array_num))[0][0].item()
            out8 = net(inputs[:,:10,:,:].unsqueeze(2).reshape(-1,1,56,array_num),inputs[:,80:90,:,:].unsqueeze(2).reshape(-1,1,56,array_num))[0][0].item()
            out9 = net(inputs[:,:10,:,:].unsqueeze(2).reshape(-1,1,56,array_num),inputs[:,90:100,:,:].unsqueeze(2).reshape(-1,1,56,array_num))[0][0].item()
            out10 = net(inputs[:,:10,:,:].unsqueeze(2).reshape(-1,1,56,array_num),inputs[:,100:110,:,:].unsqueeze(2).reshape(-1,1,56,array_num))[0][0].item()
            # print(labels)
            loc1 = np.sqrt((labels[0]-labels[3])**2+(labels[1]-labels[4])**2+(labels[2]-labels[5])**2).item()
            loc2 = np.sqrt((labels[0]-labels[6])**2+(labels[1]-labels[7])**2+(labels[2]-labels[8])**2).item()
            loc3 = np.sqrt((labels[0]-labels[9])**2+(labels[1]-labels[10])**2+(labels[2]-labels[11])**2).item()
            loc4 = np.sqrt((labels[0]-labels[12])**2+(labels[1]-labels[13])**2+(labels[2]-labels[14])**2).item()
            loc5 = np.sqrt((labels[0]-labels[15])**2+(labels[1]-labels[16])**2+(labels[2]-labels[17])**2).item()
            loc6 = np.sqrt((labels[0]-labels[18])**2+(labels[1]-labels[19])**2+(labels[2]-labels[20])**2).item()
            loc7 = np.sqrt((labels[0]-labels[21])**2+(labels[1]-labels[22])**2+(labels[2]-labels[23])**2).item()
            loc8 = np.sqrt((labels[0]-labels[24])**2+(labels[1]-labels[25])**2+(labels[2]-labels[26])**2).item()
            loc9 = np.sqrt((labels[0]-labels[27])**2+(labels[1]-labels[28])**2+(labels[2]-labels[29])**2).item()
            loc10 = np.sqrt((labels[0]-labels[30])**2+(labels[1]-labels[31])**2+(labels[2]-labels[32])**2).item()
            
            
            outlist = np.array([out1,out2,out3,out4,out5,out6,out7,out8,out9,out10]) # larger values indicate closer proximity
            # print(outlist)
            loclist = np.array([loc1,loc2,loc3,loc4,loc5,loc6,loc7,loc8,loc9,loc10]) # Smaller values indicate closer proximity
            # print(loclist)
            for iters in range(repeat_times):
                randndcg9 = np.random.permutation(10)[:9]
                randndcg8 = np.random.permutation(10)[:8]
                randndcg7 = np.random.permutation(10)[:7]
                randndcg6 = np.random.permutation(10)[:6]
                randndcg5 = np.random.permutation(10)[:5]
                randndcg4 = np.random.permutation(10)[:4]
                randndcg3 = np.random.permutation(10)[:3]
                randndcg2 = np.random.permutation(10)[:2]
                
                ndcg9[0,iters] = ndcg9[0,iters] + LIS(rank_elements_max(outlist[randndcg9]),rank_elements_min(loclist[randndcg9]))
                ndcg8[0,iters] = ndcg8[0,iters] + LIS(rank_elements_max(outlist[randndcg8]),rank_elements_min(loclist[randndcg8]))
                ndcg7[0,iters] = ndcg7[0,iters] + LIS(rank_elements_max(outlist[randndcg7]),rank_elements_min(loclist[randndcg7]))
                ndcg6[0,iters] = ndcg6[0,iters] + LIS(rank_elements_max(outlist[randndcg6]),rank_elements_min(loclist[randndcg6]))
                ndcg5[0,iters] = ndcg5[0,iters] + LIS(rank_elements_max(outlist[randndcg5]),rank_elements_min(loclist[randndcg5]))
                ndcg4[0,iters] = ndcg4[0,iters] + LIS(rank_elements_max(outlist[randndcg4]),rank_elements_min(loclist[randndcg4]))
                ndcg3[0,iters] = ndcg3[0,iters] + LIS(rank_elements_max(outlist[randndcg3]),rank_elements_min(loclist[randndcg3]))
                ndcg2[0,iters] = ndcg2[0,iters] + LIS(rank_elements_max(outlist[randndcg2]),rank_elements_min(loclist[randndcg2]))
            
                accuracy9[0,iters] = accuracy9[0,iters] + acc(rank_elements_max(outlist[randndcg9]),rank_elements_min(loclist[randndcg9]))
                accuracy8[0,iters] = accuracy8[0,iters] + acc(rank_elements_max(outlist[randndcg8]),rank_elements_min(loclist[randndcg8]))
                accuracy7[0,iters] = accuracy7[0,iters] + acc(rank_elements_max(outlist[randndcg7]),rank_elements_min(loclist[randndcg7]))
                accuracy6[0,iters] = accuracy6[0,iters] + acc(rank_elements_max(outlist[randndcg6]),rank_elements_min(loclist[randndcg6]))
                accuracy5[0,iters] = accuracy5[0,iters] + acc(rank_elements_max(outlist[randndcg5]),rank_elements_min(loclist[randndcg5]))
                accuracy4[0,iters] = accuracy4[0,iters] + acc(rank_elements_max(outlist[randndcg4]),rank_elements_min(loclist[randndcg4]))
                accuracy3[0,iters] = accuracy3[0,iters] + acc(rank_elements_max(outlist[randndcg3]),rank_elements_min(loclist[randndcg3]))
                accuracy2[0,iters] = accuracy2[0,iters] + acc(rank_elements_max(outlist[randndcg2]),rank_elements_min(loclist[randndcg2]))
            testiter = testiter + 1
            
        # print('10 categories ndcg is：{}±{}'.format(np.mean(ndcg10/nums),np.std(ndcg10/nums)))
        print('9 categories ndcg is：{:.3f}±{:.3f}'.format(np.mean(ndcg9/nums),np.std(ndcg9/nums)))
        print('8 categories ndcg is：{:.3f}±{:.3f}'.format(np.mean(ndcg8/nums),np.std(ndcg8/nums)))
        print('7 categories ndcg is：{:.3f}±{:.3f}'.format(np.mean(ndcg7/nums),np.std(ndcg7/nums)))
        print('6 categories ndcg is：{:.3f}±{:.3f}'.format(np.mean(ndcg6/nums),np.std(ndcg6/nums)))
        print('5 categories ndcg is：{:.3f}±{:.3f}'.format(np.mean(ndcg5/nums),np.std(ndcg5/nums)))
        print('4 categories ndcg is：{:.3f}±{:.3f}'.format(np.mean(ndcg4/nums),np.std(ndcg4/nums)))
        print('3 categories ndcg is：{:.3f}±{:.3f}'.format(np.mean(ndcg3/nums),np.std(ndcg3/nums)))
        print('2 categories ndcg is：{:.3f}±{:.3f}'.format(np.mean(ndcg2/nums),np.std(ndcg2/nums)))

        # print('10 categories Top-1 accuracy is：{}±{}'.format(accuracy10/nums))
        print('9 categories Top-1 accuracy is：{:.3f}±{:.3f}'.format(np.mean(accuracy9/nums),np.std(accuracy9/nums)))
        print('8 categories Top-1 accuracy is：{:.3f}±{:.3f}'.format(np.mean(accuracy8/nums),np.std(accuracy8/nums)))
        print('7 categories Top-1 accuracy is：{:.3f}±{:.3f}'.format(np.mean(accuracy7/nums),np.std(accuracy7/nums)))
        print('6 categories Top-1 accuracy is：{:.3f}±{:.3f}'.format(np.mean(accuracy6/nums),np.std(accuracy6/nums)))
        print('5 categories Top-1 accuracy is：{:.3f}±{:.3f}'.format(np.mean(accuracy5/nums),np.std(accuracy5/nums)))
        print('4 categories Top-1 accuracy is：{:.3f}±{:.3f}'.format(np.mean(accuracy4/nums),np.std(accuracy4/nums)))
        print('3 categories Top-1 accuracy is：{:.3f}±{:.3f}'.format(np.mean(accuracy3/nums),np.std(accuracy3/nums)))
        print('2 categories Top-1 accuracy is：{:.3f}±{:.3f}'.format(np.mean(accuracy2/nums),np.std(accuracy2/nums)))
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        with open(result_dir+result_name, 'w') as f:
            # f.write('10 categories ndcg is：{}±{}\n'.format(ndcg10/nums))
            f.write('9 categories ndcg is：{:.3f}±{:.3f}\n'.format(np.mean(ndcg9/nums),np.std(ndcg9/nums)))
            f.write('8 categories ndcg is：{:.3f}±{:.3f}\n'.format(np.mean(ndcg8/nums),np.std(ndcg8/nums)))
            f.write('7 categories ndcg is：{:.3f}±{:.3f}\n'.format(np.mean(ndcg7/nums),np.std(ndcg7/nums)))
            f.write('6 categories ndcg is：{:.3f}±{:.3f}\n'.format(np.mean(ndcg6/nums),np.std(ndcg6/nums)))
            f.write('5 categories ndcg is：{:.3f}±{:.3f}\n'.format(np.mean(ndcg5/nums),np.std(ndcg5/nums)))
            f.write('4 categories ndcg is：{:.3f}±{:.3f}\n'.format(np.mean(ndcg4/nums),np.std(ndcg4/nums)))
            f.write('3 categories ndcg is：{:.3f}±{:.3f}\n'.format(np.mean(ndcg3/nums),np.std(ndcg3/nums)))
            f.write('2 categories ndcg is：{:.3f}±{:.3f}\n'.format(np.mean(ndcg2/nums),np.std(ndcg2/nums)))

            # f.write('10 categories Top-1 accuracy is：{}±{}\n'.format(accuracy10/nums))
            f.write('9 categories Top-1 accuracy is：{:.3f}±{:.3f}\n'.format(np.mean(accuracy9/nums),np.std(accuracy9/nums)))
            f.write('8 categories Top-1 accuracy is：{:.3f}±{:.3f}\n'.format(np.mean(accuracy8/nums),np.std(accuracy8/nums)))
            f.write('7 categories Top-1 accuracy is：{:.3f}±{:.3f}\n'.format(np.mean(accuracy7/nums),np.std(accuracy7/nums)))
            f.write('6 categories Top-1 accuracy is：{:.3f}±{:.3f}\n'.format(np.mean(accuracy6/nums),np.std(accuracy6/nums)))
            f.write('5 categories Top-1 accuracy is：{:.3f}±{:.3f}\n'.format(np.mean(accuracy5/nums),np.std(accuracy5/nums)))
            f.write('4 categories Top-1 accuracy is：{:.3f}±{:.3f}\n'.format(np.mean(accuracy4/nums),np.std(accuracy4/nums)))
            f.write('3 categories Top-1 accuracy is：{:.3f}±{:.3f}\n'.format(np.mean(accuracy3/nums),np.std(accuracy3/nums)))
            f.write('2 categories Top-1 accuracy is：{:.3f}±{:.3f}\n'.format(np.mean(accuracy2/nums),np.std(accuracy2/nums)))
    return

if __name__ == '__main__':
    parser = ArgumentParser(
        description='runs inference process based on trained model')
    parser.add_argument('--device', default='cuda:2',
                        help='device for inference')
    parser.add_argument('--model_dir', default='./model/',
                        help='directory in which to store model checkpoints.')
    parser.add_argument('--dataset_dir', default='./dataset/test/',
                        help='directories from which to load the test data.')
    parser.add_argument('--result_dir', default='./result/',
                        help='directories from which to store the result.')
    parser.add_argument('--model_name', default='base111.pt',
                        help='Name of the model checkpoints.')
    parser.add_argument('--dataset_name', default='test111snr30.mat',
                        help='Name of the test data.')
    
    test(parser.parse_args())
