import argparse
import os 
import numpy as np
import tensorflow as tf
import pandas as pd 

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Attack')
	parser.add_argument('--train',type=bool,default=False,help='Training')
	parser.add_argument('--train_1',type=bool,default=False,help='Training')
	parser.add_argument('--batch_size', type=int, default=1000, help='Training batch size')
	parser.add_argument('--trade_off', type=float, default=0.1, help='Trade off term')
	parser.add_argument('--mapping_dim', type=int ,default=400)
	parser.add_argument('--mapping_dim_pca', type=int ,default=130)
	parser.add_argument('--gamma', type=float, default=0.001)
	parser.add_argument('--seed', type=float, default=9.0)
	parser.add_argument('--epoch', type=int, default=20, help='Training epcohs')
	args = parser.parse_args()
	print(args)

	if args.train: 
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="0"
		import ran
		model = ran.RAN(args)
		acc, mse, mse_lrr, mse_krr  = model.train()

		'''
		count = 0.01
		para = []
		#for i in range(90):
		for i in range(90):
			para.append(np.round(count, 2))
			count += 0.01

		acc_list = []
		mse_list = []
		mse_lrr_list = []
		mse_krr_list = [] 


		for i in para: 
			
			print("***************************************")
			args.trade_off = i
			print(args)
			tf.reset_default_graph()
			model = ran.RAN(args)
			acc, mse, mse_lrr, mse_krr  = model.train()
			acc_list.append(acc)
			mse_list.append(mse)
			mse_lrr_list.append(mse_lrr)
			mse_krr_list.append(mse_krr)
			print("***************************************")

		Matrix = {}
		Matrix['Lambda'] = para
		Matrix['acc']= acc_list
		Matrix['mse_nn'] = mse_list
		Matrix['mse_lrr'] = mse_lrr_list
		Matrix['mse_krr'] = mse_krr_list
		final = pd.DataFrame(Matrix)
		final.to_csv("ran_new_v.csv", index=False)
		'''
		
	elif args.train_1 :
		
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="1"
		import ran
		count = 0.01
		para = []
		#for i in range(90):
		for i in range(90):
			para.append(np.round(count, 2))
			count += 0.01

		acc_list = []
		mse_list = []
		mse_lrr_list = []
		mse_krr_list = [] 

		for i in para: 
			print("***************************************")
			args.trade_off = i
			print(args)
			tf.reset_default_graph()
			model = ran.RAN(args)
			acc, mse, mse_lrr, mse_krr  = model.train()
			acc_list.append(acc)
			mse_list.append(mse)
			mse_lrr_list.append(mse_lrr)
			mse_krr_list.append(mse_krr)
			print("***************************************")
		Matrix = {}
		Matrix['Lambda'] = para
		Matrix['acc']= acc_list
		Matrix['mse_nn'] = mse_list
		Matrix['mse_lrr'] = mse_lrr_list
		Matrix['mse_krr'] = mse_krr_list
		final = pd.DataFrame(Matrix)
		final.to_csv("ran_new_v.csv", index=False)

	else:
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="1"
		import ran
		model = ran.RAN(args)
		model.train()