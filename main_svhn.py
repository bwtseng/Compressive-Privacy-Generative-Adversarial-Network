import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import h5py 
from functools import reduce
import tensorflow as tf
import tensorflow.contrib.layers as ly 
import time
from scipy.misc import imread, imresize
import argparse
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment_1')
    parser.add_argument('--train',type=bool,default=False,help='Training')
    #parser.add_argument('--path',type=str,default='Resize_64',help='Path for your face')
    parser.add_argument('--test',type=bool,default=False,help='Testing')
    parser.add_argument('--model_path',type=str,default='path/to/your/mopdel',help='Model Path')
    #parser.add_argument('--train_1',type=bool,default=False,help='Training')
    args = parser.parse_args()
    print(args)

    if args.train : 
        import svhn
        model = svhn.wrs(args)
        model.train() 
    else : 
        import svhn
        model = svhn.wrs(args)
        model.test()            



    '''
    if args.train_0 :
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        import svhn
        model = svhn.shakenet(args)
        model.train()
    elif args.train_1: 
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]="1"
        import svhn
        model = svhn.shakenet(args)
        model.train()
    else : 
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]="1"
        import svhn
        model = svhn.shakenet(args)  
        model.test()
    '''