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
    parser = argparse.ArgumentParser(description='SVHN classification task')
    parser.add_argument('--train_1', type=bool, default=False, help='Training with gpu:1')
    parser.add_argument('--train_0', type=bool, default=False, help='Training with gpu:1')
    parser.add_argument('--test', type=bool,default=False,help='Testing')
    parser.add_argument('--path', type=str,default='/path/to/mat/file', help='Path to your SVHN')
    parser.add_argument('--cut_out', type=bool,default=False,help='cut_out')
    parser.add_argument('--model_path', type=str, default='/path/to/model/file', help='Load model')
    args = parser.parse_args()
    print(args)

    if args.train_0 :
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        import wrs_svhn
        model = wrs_svhn.wrs(args)
        model.train()

    elif args.train_1: 
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]="1"
        import wrs_svhn
        model = wrs_svhn.wrs(args)
        model.train()
    else : 
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        import wrs_svhn
        model = wrs_svhn.wrs(args)
        model.test()