import argparse
import os 
import numpy as np
import pandas as pd 
import tensorflow as tf 


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Attack')
	parser.add_argument('--train',type=bool,default=False,help='Training')
	parser.add_argument('--train_1', type=bool, default=False)
	parser.add_argument('--ori_dim', type=int, default=32, help='Orignal dimension')
	parser.add_argument('--com_dim', type=int, default=16, help='Compressive dimension')
	parser.add_argument('--prior_prob', type=float, default=0.5, help='Prior probability')
	parser.add_argument('--batch_size', type=int, default=1000, help='Training batch size')
	parser.add_argument('--samples', type=int, default=22000, help='Training data size')
	parser.add_argument('--trade_off', type=float, default=0.1, help='Trade off term')
	parser.add_argument('--noise_term', type=float, default=0.1, help='Noise factorsss')
	parser.add_argument('--epoch', type=int, default=100, help='Training epcohs')
	parser.add_argument('--seed', type=int, default=9)
	parser.add_argument('--gamma', type=float, default=0.0001)
	parser.add_argument('--mapping_dim',type=int, default=5000)
	args = parser.parse_args()
	print(args)

	if args.train: 
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="0"
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
		theory_acc_list = []
		theory_mse_list = []

		loop = 1
		for i in para: 
			print("***************************************")
			args.trade_off = i
			print(args)
			tf.reset_default_graph()
			model = ran.RAN(args)
			acc, mse, theory_acc, theory_mse, mse_lrr, mse_krr  = model.train()
			acc_list.append(acc)
			mse_list.append(mse)
			theory_acc_list.append(theory_acc)
			theory_mse_list.append(theory_mse)
			mse_lrr_list.append(mse_lrr)
			mse_krr_list.append(mse_krr)
			print("***************************************")

			if loop %10 ==0: 
				Matrix = {}
				Matrix['Lambda'] = para
				Matrix['acc']= acc_list
				Matrix['acc_theory'] = theory_acc_list
				Matrix['mse_nn'] = mse_list
				Matrix['mse_lrr'] = mse_lrr_list
				Matrix['mse_krr'] = mse_krr_list
				Matrix['theory_mse_list'] = theory_mse_list
				final = pd.DataFrame(Matrix)
				final.to_csv("ran.csv", index=False)

			loop += 1 

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
		theory_acc_list = []
		theory_mse_list = []
		lamda_list = []
		loop = 1
		for i in para: 
			print("***************************************")
			args.trade_off = i
			print(args)
			tf.reset_default_graph()
			model = ran.RAN(args)
			acc, mse, theory_acc, theory_mse, mse_lrr, mse_krr  = model.train()
			acc_list.append(acc)
			mse_list.append(mse)
			theory_acc_list.append(theory_acc)
			theory_mse_list.append(theory_mse)
			mse_lrr_list.append(mse_lrr)
			mse_krr_list.append(mse_krr)
			lamda_list.append(i)
			print("***************************************")

			if loop % 10 ==0: 
				Matrix = {}
				Matrix['acc']= acc_list
				Matrix['acc_theory'] = theory_acc_list
				Matrix['mse_nn'] = mse_list
				Matrix['mse_lrr'] = mse_lrr_list
				Matrix['mse_krr'] = mse_krr_list
				Matrix['Lambda'] = lamda_list
				Matrix['theory_mse_list'] = theory_mse_list
				final = pd.DataFrame(Matrix)
				final.to_csv("ran.csv", index=False)

			loop += 1

	else:
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="1"
		import ran
		model = ran.RAN(args)
		model.train()