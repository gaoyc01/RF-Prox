import numpy as np
import h5py

def get_data(filename, array_num):
# load and construct datase
    data = h5py.File(filename)
    ref = data['datasets']
    num = ref.shape[0]
    dataset = np.zeros([num,10,56,array_num],dtype=complex)
    locs = np.zeros([num,3])
    for i in range(num):
        dataset[i,:,:,:] = np.transpose(np.transpose(data[ref[i,0]]['CSI'])['real'],[0,2,1]) + 1j * np.transpose(np.transpose(data[ref[i,0]]['CSI'])['imag'],[0,2,1])
        locs[i,:] = np.transpose(data[ref[i,0]]['location'])

    num_all = len(dataset)
    print(dataset.shape)
    print(locs.shape)
    
    # shuffle dataset
    np.random.seed(1)
    permutation = np.random.permutation(num_all)
    dataset = dataset[permutation]
    locs = locs[permutation]

    train_num = 60000
    np.random.seed(2)
    trainrand = np.random.randint(0,num_all, (2,train_num))
    traindata = np.concatenate((dataset[trainrand[0]],dataset[trainrand[1]]),axis=1) # (20000, 20, 64, 64)
    print(traindata.shape)
    trainloc = np.concatenate((locs[trainrand[0]],locs[trainrand[1]]),axis=1) # (20000, 6)
    print(trainloc.shape)
    trainlocs = np.zeros([1,train_num])
    for i in range(train_num):
        trainlocs[0,i] = np.sqrt((trainloc[i,0]-trainloc[i,3])**2 + (trainloc[i,1]-trainloc[i,4])**2 + (trainloc[i,2]-trainloc[i,5])**2)
    print(trainlocs.shape)
    return traindata,trainlocs

def get_data_test(filename, array_num):
    # load and construct datase
    data = h5py.File(filename)
    ref = data['datasets']
    num = ref.shape[0]
    dataset = np.zeros([num,10,56,array_num],dtype=complex)
    locs = np.zeros([num,3])
    for i in range(num):
        dataset[i,:,:,:] = np.transpose(np.transpose(data[ref[i,0]]['CSI'])['real'],[0,2,1]) + 1j * np.transpose(np.transpose(data[ref[i,0]]['CSI'])['imag'],[0,2,1])
        locs[i,:] = np.transpose(data[ref[i,0]]['location'])

    # concatenate dataset
    num_all = len(dataset)
    print(dataset.shape)
    print(locs.shape)
    
    # shuffle dataset
    permutation = np.random.permutation(num_all)
    dataset = dataset[permutation]
    locs = locs[permutation]
    
    return dataset,locs


def get_data_fine(filename, array_num):
# load and construct datase
    data = h5py.File(filename)
    ref = data['datasets']
    num = ref.shape[0]
    dataset = np.zeros([num,10,56,array_num],dtype=complex)
    locs = np.zeros([num,3])
    for i in range(num):
        dataset[i,:,:,:] = np.transpose(np.transpose(data[ref[i,0]]['CSI'])['real'],[0,2,1]) + 1j * np.transpose(np.transpose(data[ref[i,0]]['CSI'])['imag'],[0,2,1])
        locs[i,:] = np.transpose(data[ref[i,0]]['location'])

    num_all = dataset.shape[0]
    print(dataset.shape)
    print(locs.shape)
    
    # shuffle dataset
    np.random.seed(1)
    permutation = np.random.permutation(num_all)
    dataset = dataset[permutation]
    locs = locs[permutation]

    
    train_num = 10000
    num_all = 1000
    
    dataset = dataset[:num_all]
    locs = locs[:num_all]
    print(dataset.shape)
    print(locs.shape)
    
    np.random.seed(2)
    trainrand = np.random.randint(0,num_all, (2,train_num))
    traindata = np.concatenate((dataset[trainrand[0]],dataset[trainrand[1]]),axis=1) # (20000, 20, 64, 64)
    print(traindata.shape)
    trainloc = np.concatenate((locs[trainrand[0]],locs[trainrand[1]]),axis=1) # (20000, 6)
    print(trainloc.shape)
    trainlocs = np.zeros([1,train_num])
    for i in range(train_num):
        trainlocs[0,i] = np.sqrt((trainloc[i,0]-trainloc[i,3])**2 + (trainloc[i,1]-trainloc[i,4])**2 + (trainloc[i,2]-trainloc[i,5])**2)
    print(trainlocs.shape)
    return traindata,trainlocs