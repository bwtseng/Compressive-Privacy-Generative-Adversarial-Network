import argparse
import os 
import numpy as np 
import tensorflow as tf 
import pandas as pd 

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Attack')
	parser.add_argument('--path', type=str, default='/home/pywu/bowei/Genki/datasets')
	parser.add_argument('--train', type=bool, default=False, help='Training')
	parser.add_argument('--train_1', type=bool, default=False)
	parser.add_argument('--batch_size', type=int, default=512, help='Training batch size')
	parser.add_argument('--noise_term', type=float, default=0.1, help='Noise factorsss')
	parser.add_argument('--noise_scale', type=float, default=1) #5)
	parser.add_argument('--pca_dim', type=int, default=200)
	parser.add_argument('--mapping_dim', type=int, default=5000)
	parser.add_argument('--mapping_dim_pca', type=int ,default=128)
	parser.add_argument('--gamma', type=float, default=0.001)
	parser.add_argument('--seed', type=float, default=9.0)
	parser.add_argument('--epoch', type=int, default=50, help='Training epcohs')
	args = parser.parse_args()
	print(args)

	if args.train: 
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="0"
		import dnn
		model =  dnn.DNN(args)
		acc, mse, mse_lrr, mse_krr, acc_pca, mse_pca, mse_pca_lrr, mse_pca_krr = model.train()
		'''
		para = [] 
		count = 0.1 
		for i in range(17): 
			para.append(np.round(count, 2))
			count += 0.05 
		
		#para = [0.9]

		acc_list = []
		mse_list = []
		mse_list_lrr = []
		mse_list_krr = []
		acc_pca_list = []
		mse_pca_list = []
		mse_pca_list_lrr = []
		mse_pca_list_krr = []

		for i in para: 
			print("***************************************")
			args.noise_term = i
			print(args)
			tf.reset_default_graph()
			model =  dnn.DNN(args)
			acc, mse, mse_lrr, mse_krr, acc_pca, mse_pca, mse_pca_lrr, mse_pca_krr = model.train()
			acc_list.append(acc)
			mse_list.append(mse)
			mse_list_lrr.append(mse_lrr)
			mse_list_krr.append(mse_krr)

			acc_pca_list.append(acc_pca)
			mse_pca_list.append(mse_pca)
			mse_pca_list_lrr.append(mse_pca_lrr)
			mse_pca_list_krr.append(mse_pca_krr)
			#noise_term_list.append(i)
			print("***************************************")

		Matrix = {}
		Matrix['Noise'] = para
		Matrix['acc']= acc_list
		Matrix['mse'] = mse_list
		Matrix['mse_lrr'] = mse_list_lrr
		Matrix['mse_krr']= mse_list_krr
		Matrix['acc_pca'] = acc_pca_list

		Matrix['mse_pca'] = mse_pca_list
		Matrix['mse_pca_lrr']= mse_pca_list_lrr
		Matrix['mse_pca_krr'] = mse_pca_list_krr

		final = pd.DataFrame(Matrix)
		final.to_csv("dnn.csv", index=False)
		'''

	elif args.train_1:
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="1"
		import dnn
		
		para = [] 
		count = 0.1 
		for i in range(17): 
			para.append(np.round(count, 2))
			count += 0.05 
		
		#para = [0.1]

		acc_list = []
		mse_list = []
		mse_list_lrr = []
		mse_list_krr = []
		acc_pca_list = []
		mse_pca_list = []
		mse_pca_list_lrr = []
		mse_pca_list_krr = []

		for i in para: 
			print("***************************************")
			args.noise_term = i
			print(args)
			tf.reset_default_graph()
			model =  dnn.DNN(args)
			acc, mse, mse_lrr, mse_krr, acc_pca, mse_pca, mse_pca_lrr, mse_pca_krr = model.train()
			acc_list.append(acc)
			mse_list.append(mse)
			mse_list_lrr.append(mse_lrr)
			mse_list_krr.append(mse_krr)

			acc_pca_list.append(acc_pca)
			mse_pca_list.append(mse_pca)
			mse_pca_list_lrr.append(mse_pca_lrr)
			mse_pca_list_krr.append(mse_pca_krr)
			#noise_term_list.append(i)
			print("***************************************")

		Matrix = {}
		Matrix['Noise'] = para
		Matrix['acc']= acc_list
		Matrix['mse'] = mse_list
		Matrix['mse_lrr'] = mse_list_lrr
		Matrix['mse_krr']= mse_list_krr
		Matrix['acc_pca'] = acc_pca_list

		Matrix['mse_pca'] = mse_pca_list
		Matrix['mse_pca_lrr']= mse_pca_list_lrr
		Matrix['mse_pca_krr'] = mse_pca_list_krr

		final = pd.DataFrame(Matrix)
		final.to_csv("dnn.csv", index=False)

	else:
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="1"
		import dnn
		model = dnn.DNN(args)
		model.train()