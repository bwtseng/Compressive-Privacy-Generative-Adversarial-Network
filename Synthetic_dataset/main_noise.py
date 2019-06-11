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
	parser.add_argument('--noise_term', type=float, default=0.0001, help='Noise factors')
	parser.add_argument('--epoch', type=int, default=100, help='Training epcohs')
	args = parser.parse_args()
	print(args)

	if args.train: 
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="0"
		import theory_noise
		model = theory_noise.syn_noise(args)
		model.train()

	elif args.train_1 :
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="1"
		import theory_noise as tn
		para = [] 
		count = 0.1 
		for i in range(17):
			para.append(count)
			count += 0.05 

		acc_list = []
		mse_list = []
		noise_term_list = []
		for i in para: 
			print("***************************************")
			args.noise_term = i
			print(args)
			tf.reset_default_graph()
			model =  tn.syn_noise(args)
			acc, mse = model.train()
			acc_list.append(acc)
			mse_list.append(mse)
			noise_term_list.append(i)
			print("***************************************")

		Matrix = {}
		Matrix['Noise'] = para
		Matrix['acc']= acc_list
		Matrix['mse'] = mse_list
		final = pd.DataFrame(Matrix)
		final.to_csv("noise.csv", index=False)


	else:
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="1"
		import theory_noise
		model = theory_noise.syn_noise(args)
		model.train()