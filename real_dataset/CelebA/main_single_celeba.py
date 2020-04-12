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
    parser = argparse.ArgumentParser(description='single_task_CelebA')
    #parser.add_argument('--train_0', default=False, action='store_true', help='Training with gpu:0')
    #parser.add_argument('--train_1', default=False, action='store_true', help='Training with gpu:1')
    parser.add_argument('--train', default=False, action='store_true', help='Training phase' )
    parser.add_argument('--test',  default=False, action='store_true', help='Testing phase')
    parser.add_argument('--data_dir',type=str,default='/path/to/your/celeba_dataset', help='Path to your facial images.')
    parser.add_argument('--label_dir', type=str, default='/path/to/your/label_file', help='Path to your label csv file.')
    parser.add_argument('--model_dir', type=str, default='/path/to/your/trained_model', help='Pre-trained model path.')
    parser.add_argument('--attribute', '-a', type=str, default='Male', help='Which attributes you specify')
    parser.add_argument('--com_dim', type=int, default=2, help='Compressive dimension')
    parser.add_argument('--batch_size', '-b', type=int, default=1024, help='Batch_size')
    parser.add_argument('--epoch', type=int, default=15, help='Training epoch')
    parser.add_argument('--citer', type=int ,default=25, help='Adverary training iteration.')
    parser.add_argument('--name', type=str, default='single-task-celeba', help='name of the checkpoint.')
    args = parser.parse_args()
    print(args)

    if args.train:
        import celeba_0
        model = celeba_0.CPGAN(args)
        model.train()
    elif args.test: 
        import celeba_0
        model = celeba_0.CPGAN(args)     
        model.test()
    else:
    	raise ValueError("Plear input correct args !!!")
    # ****************
    # For GPU control.
    # ****************
    """
    if args.train_0 :
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        import celeba_0
        model = celeba_0.cpgan(args)
        model.train()

    elif args.train_1: 
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]="1"
        import celeba_0
        model = celeba_0.cpgan(args)
        model.train()
    else : 
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        import celeba_0
        model = celeba_0.cpgan(args)     
        model.test()
    """