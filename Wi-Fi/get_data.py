import numpy as np
import h5py

def get_data(filename, array_num):
# load and construct dataset
    filename1 = filename
    data1 = h5py.File(filename1)
    ref1 = data1['dataCell']
    num1 = ref1.shape[1] # 30
    numUE = 20
    numIoT = 5
    times = 11
    num_percell = numUE*numIoT

    dataset1 = np.zeros([num1*num_percell,times*2,56,array_num],dtype=complex)
    dist1 = np.zeros([num1*num_percell,1])
    # print(np.transpose(data1[ref1[0,0]])['real'].shape) #(5, 11, 56, 9)
    index = 0
    for i in range(num1):
        dataIoT = np.transpose(data1[ref1[0,i]])['real'] + 1j * np.transpose(data1[ref1[0,i]])['imag'] #(5, 11, 56, 9)
        dataUE = np.transpose(data1[ref1[1,i]])['real'] + 1j * np.transpose(data1[ref1[1,i]])['imag'] #(20, 11, 56, 9)
        distUEIoT = np.transpose(data1[ref1[2,i]]) # (5,20)
        for k in range(numIoT):
            for kk in range(numUE):
                dataset1[index,:,:,:] = np.concatenate((dataIoT[k,:,:,:],dataUE[kk,:,:,:]),axis = 0)
                dist1[index,0] = distUEIoT[kk,k]
                index = index + 1

    dataset = dataset1
    dist = dist1
    return dataset,dist

def get_data_test(filename, array_num):
    # load and construct dataset
    filename1 = filename
    data1 = h5py.File(filename1)
    ref1 = data1['dataCell']
    num1 = ref1.shape[1] # 30
    # num1 = 10
    numUE = 100
    numIoT = 10
    times = 11
    num_percell = numUE

    dataset1 = np.zeros([num1*num_percell,times*(numIoT+1),56,array_num],dtype=complex)
    dist1 = np.zeros([num1*num_percell,numIoT])

    index = 0
    for i in range(num1):
        dataIoT = np.transpose(data1[ref1[0,i]])['real'] + 1j * np.transpose(data1[ref1[0,i]])['imag'] #(5, 11, 56, 9)
        dataUE = np.transpose(data1[ref1[1,i]])['real'] + 1j * np.transpose(data1[ref1[1,i]])['imag'] #(20, 11, 56, 9)
        distUEIoT = np.transpose(data1[ref1[2,i]]) # (20,5)
        # print(distUEIoT.shape)
        for k in range(numUE):
            dataset1[index,:,:,:] = np.concatenate((dataUE[k,:,:,:],dataIoT[0,:,:,:],dataIoT[1,:,:,:],dataIoT[2,:,:,:],dataIoT[3,:,:,:],dataIoT[4,:,:,:],\
                                                    dataIoT[5,:,:,:],dataIoT[6,:,:,:],dataIoT[7,:,:,:],dataIoT[8,:,:,:],dataIoT[9,:,:,:]),axis = 0)
            dist1[index,:] = np.array([distUEIoT[k,0],distUEIoT[k,1],distUEIoT[k,2],distUEIoT[k,3],distUEIoT[k,4],\
                                       distUEIoT[k,5],distUEIoT[k,6],distUEIoT[k,7],distUEIoT[k,8],distUEIoT[k,9]])
            index = index + 1

    dataset = dataset1
    dist = dist1
    return dataset,dist
    data1 = h5py.File(filename)
    ref1 = data1['dataset']
    num1 = ref1.shape[0]
    dataset = np.zeros([num1,11,56,array_num],dtype=complex)
    locs = np.zeros([num1,3])

    for i in range(num1):
        dataset[i,:,:,:] = np.transpose(np.transpose(data1[ref1[i,0]]['CSI'])['real'], [0,2,1]) + 1j * np.transpose(np.transpose(data1[ref1[i,0]]['CSI'])['imag'], [0,2,1])
        locs[i,:] = np.transpose(data1[ref1[i,0]]['location'])

    # concatenate dataset
    np.random.seed(1)
    num_all = len(dataset)
    # shuffle dataset
    permutation = np.random.permutation(num_all)
    dataset = dataset[permutation]
    locs = locs[permutation]

    # split dataset
    np.random.seed(3)
    testrand = np.arange(num_all)
    testdata = dataset[testrand] # (2678, 10, 64, 64)
    
    # normalization
    min_val = np.min(testdata)
    max_val = np.max(testdata)

    # 缩放数据
    testdata = 2 * (testdata - min_val) / (max_val - min_val) - 1

    testloc = locs[testrand] # (2678, 3)
    return testdata,testloc