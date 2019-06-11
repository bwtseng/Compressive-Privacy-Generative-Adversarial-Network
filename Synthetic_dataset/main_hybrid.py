import argparse
import os 
import numpy as np
import tensorflow as tf 
import pandas as pd 

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='Attack')
	parser.add_argument('--train',type=bool,default=False,help='Training')
	parser.add_argument('--train_1', type=bool, default=False)
	parser.add_argument('--citer', type=int, default=25)
	parser.add_argument('--ori_dim', type=int, default=32, help='Orignal dimension')
	parser.add_argument('--com_dim', type=int, default=12, help='Compressive dimension') #16
	parser.add_argument('--prior_prob', type=float, default=0.5, help='Prior probability')
	parser.add_argument('--batch_size', type=int, default=1000, help='Training batch size')
	parser.add_argument('--samples', type=int, default=22000, help='Training data size')
	parser.add_argument('--trade_off', type=float, default=0.1, help='Trade off term')
	parser.add_argument('--noise_term', type=float, default=0.001, help='Noise factorsss')
	parser.add_argument('--epoch', type=int, default=200, help='Training epcohs')
	parser.add_argument('--seed', type=int, default=9)
	parser.add_argument('--gamma', type=float, default=0.001)
	parser.add_argument('--mapping_dim',type=int, default=5000)
	parser.add_argument('--pca_dim', type=int, default=8)
	args = parser.parse_args()
	print(args)

	if args.train: 

		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="0"

		import hybrid_cpgan
		count = 1
		para = []
		for i in range(89):
		#for i in range(1):
			#para.append(np.round(count, 2))
			para.append(count)
			count += 1
		### if lambd is too small, CPGAN is unable to learn persist attack.

		para.append(100)

		acc_list = []
		mse_list = []
		mse_lrr_list = []
		mse_krr_list = [] 
		theory_acc_list = []
		theory_mse_list = []
		lamda_list = []
		loops = 1
		for i in para: 

			print("***************************************")
			args.trade_off = i
			print(args)
			tf.reset_default_graph()
			model = hybrid_cpgan.hybrid_CPGAN(args)
			acc, mse, theory_acc, theory_mse, mse_lrr, mse_krr  = model.train()
			acc_list.append(acc)
			mse_list.append(mse)
			theory_acc_list.append(theory_acc)
			theory_mse_list.append(theory_mse)
			mse_lrr_list.append(mse_lrr)
			mse_krr_list.append(mse_krr)
			lamda_list.append(i)
			print("***************************************")

			if loops % 10 == 0 :
				Matrix = {}
				Matrix["Lambda"] = lamda_list
				Matrix['acc']= acc_list
				Matrix['acc_theory'] = theory_acc_list
				Matrix['mse_nn'] = mse_list
				Matrix['mse_lrr'] = mse_lrr_list
				Matrix['mse_krr'] = mse_krr_list
				Matrix['theory_mse_list'] = theory_mse_list
				final = pd.DataFrame(Matrix)
				final.to_csv("hybrid_cpgan_new_v_2.csv", index=False)
			loops+=1

	elif args.train_1 :

		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="1"

		import hybrid_cpgan
		count = 1
		para = []
		for i in range(89):
		#for i in range(1):
			#para.append(np.round(count, 2))
			para.append(count)
			count += 1
		### if lambd is too small, CPGAN is unable to learn persist attack.

		para.append(100)


		acc_list = []
		mse_list = []
		mse_lrr_list = []
		mse_krr_list = [] 
		theory_acc_list = []
		theory_mse_list = []

		for i in para: 

			print("***************************************")
			args.trade_off = i
			print(args)
			tf.reset_default_graph()
			model = hybrid_cpgan.hybrid_CPGAN(args)
			acc, mse, theory_acc, theory_mse, mse_lrr, mse_krr  = model.train()
			acc_list.append(acc)
			mse_list.append(mse)
			theory_acc_list.append(theory_acc)
			theory_mse_list.append(theory_mse)
			mse_lrr_list.append(mse_lrr)
			mse_krr_list.append(mse_krr)
			print("***************************************")

		Matrix = {}
		Matrix['Lambda'] = para
		Matrix['acc']= acc_list
		Matrix['acc_theory'] = theory_acc_list
		Matrix['mse_nn'] = mse_list
		Matrix['mse_lrr'] = mse_lrr_list
		Matrix['mse_krr'] = mse_krr_list
		Matrix['theory_mse_list'] = theory_mse_list
		final = pd.DataFrame(Matrix)
		final.to_csv("hybrid_cpgan_new_v.csv", index=False)


	else:

		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="1"
		import hybrid_cpgan
		model = hybrid_cpgan.hybrid_CPGAN(args)
		model.train()
