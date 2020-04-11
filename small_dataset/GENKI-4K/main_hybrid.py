import argparse
import os 
import numpy as np
import pandas as pd 
import tensorflow as tf 

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Attack')
	parser.add_argument('--path', type=str, default='/home/pywu/bowei/Genki/datasets')
	parser.add_argument('--com_dim', type=int, default=2048) # 3136 , 1024
	parser.add_argument('--train',type=bool,default=False,help='Training')
	parser.add_argument('--train_1',type=bool,default=False,help='Training')
	parser.add_argument('--batch_size', type=int, default=512, help='Training batch size')
	parser.add_argument('--trade_off', type=float, default=1.0, help='Trade off term')
	parser.add_argument('--mapping_dim', type=int ,default=2048)
	parser.add_argument('--gamma', type=float, default=0.001)
	parser.add_argument('--seed', type=float, default=9.0)
	parser.add_argument('--epoch', type=int, default=50, help='Training epcohs')
	parser.add_argument('--citer', type=int, default=5) ##. 3 is the best
	args = parser.parse_args()
	print(args)

	if args.train: 
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="0"

		import hybrid_cpgan as cp
		
		model = cp.hybrid_CPGAN(args)
		acc, mse, mse_lrr, mse_krr  = model.train()

		'''
		#para = [0.001, 0.01, 0.1]
		para = []
		count = 1
		for i in range(90): #40
			para.append(count)
			count += 1

		#para.append(90)
		para.append(100)
		
		#para = [100]
		acc_list = []
		mse_list = []
		mse_lrr_list = []
		mse_krr_list = [] 
		lambda_list = []
		loop = 1 
		for i in para: 
			print("***************************************")
			args.trade_off = i
			print(args)
			tf.reset_default_graph()
			model = cp.hybrid_CPGAN(args)
			acc, mse, mse_lrr, mse_krr  = model.train()

			if acc > 0.75 :
				acc_list.append(acc)
				mse_list.append(mse)
				mse_lrr_list.append(mse_lrr)
				mse_krr_list.append(mse_krr)
				lambda_list.append(i)
			print("***************************************")

			if loop % 10 == 0 : 
				Matrix = {}
				Matrix['Lambda'] = lambda_list
				Matrix['acc']= acc_list
				Matrix['mse_nn'] = mse_list
				Matrix['mse_lrr'] = mse_lrr_list
				Matrix['mse_krr'] = mse_krr_list
				final = pd.DataFrame(Matrix)
				final.to_csv("hybrid_cpgan_nonlinear.csv", index=False)

			loop += 1 
		
		Matrix = {}
		Matrix['Lambda'] = lambda_list
		Matrix['acc']= acc_list
		Matrix['mse_nn'] = mse_list
		Matrix['mse_lrr'] = mse_lrr_list
		Matrix['mse_krr'] = mse_krr_list
		final = pd.DataFrame(Matrix)
		final.to_csv("hybrid_cpgan_nonlinear.csv", index=False)	
		'''
	elif args.train_1 :
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="1"

		import hybrid_cpgan as cp
		'''
		para = [0.001, 0.01, 0.1]
		count = 1
		for i in range(85): #40
			para.append(count)
			count += 1

		#para.append(90)
		para.append(100)
		'''
		para = [1]
		acc_list = []
		mse_list = []
		mse_lrr_list = []
		mse_krr_list = [] 
		lambda_list = []
		loop = 1 
		for i in para: 
			print("***************************************")
			args.trade_off = i
			print(args)
			tf.reset_default_graph()
			model = cp.hybrid_CPGAN(args)
			acc, mse, mse_lrr, mse_krr  = model.train()
			acc_list.append(acc)
			mse_list.append(mse)
			mse_lrr_list.append(mse_lrr)
			mse_krr_list.append(mse_krr)
			lambda_list.append(i)
			print("***************************************")

			if loop % 10 == 0 : 
				Matrix = {}
				Matrix['Lambda'] = lambda_list
				Matrix['acc']= acc_list
				Matrix['mse_nn'] = mse_list
				Matrix['mse_lrr'] = mse_lrr_list
				Matrix['mse_krr'] = mse_krr_list
				final = pd.DataFrame(Matrix)
				final.to_csv("hybrid_cpgan.csv", index=False)
			loop += 1 


	else:
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="1"
		import hybrid_cpgan as cp
		model = cp.hybrid_CPGAN(args)
		model.train()