# RF-Prox
## 1. Dataset Download
Test dataset can be downloaded in [LTE](https://drive.google.com/file/d/1vfA0VDfwlNz3N2Tkx-OWUgC3tMxbePAb/view?usp=drive_link) and [Wi-Fi](https://drive.google.com/file/d/1nz7ze3H9hyGeZg0XkvdNH2uIABYDPPqk/view?usp=drive_link). 

After completing download, create a ```dataset``` folder in the root directory of the LTE or Wi-Fi folder, and create folder of ```train```, ```test```, and ```finetune```, and move the downloaded test dataset to ```LTE/dataset/test``` and ```Wi-Fi/dataset/test```, respectively.
## 2. Test with Dataset
To validate our experimental results, run ```python3 inference.py```.

The optional parameters are listed below:
```
'--device', default='cuda:2', help='device for inference')
'--model_dir', default='./model/', help='directory in which to store model checkpoints.')
'--dataset_dir', default='./dataset/test/', help='directories from which to load the test data.')
'--result_dir', default='./result/', help='directories from which to store the result.')
'--model_name', default='base111.pt', help='Name of the model checkpoints.')
'--dataset_name', default='test111snr30.mat', help='Name of the test data.')
```
## 3. Train, finetune and test with your dataset.
Move your dataset to ```LTE/dataset``` or ```Wi-Fi/dataset```.

Make sure your dataset is the shape of 
* input:([batchsize*temporal_size, 1, subcarrier, radio chains],[batchsize*temporal_size, 1, subcarrier, radio chains]).
* label:([batchsize, 1]), the shortest non-blocking distance (SNBD) between two non-directly connected devices after transformed by $p(SNBD) = tanh(−log(α \cdot SNBD))$.

First, run ```python3 train.py``` to train the model.

Then, if you want to fine-tune with the target domain dataset, run ```python3 finetune_train.py```.

Finally, run ```python3 inference.py``` to test the performance of you trained model.
