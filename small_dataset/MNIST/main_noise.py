import argparse
import os 
import numpy as np 
import pandas as pd 
import tensorflow as tf 

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='Attack')
	parser.add_argument('--train_1',type=bool,default=False,help='Training')
	parser.add_argument('--train',type=bool,default=False,help='Training')
	parser.add_argument('--noise_scale', type=float, default= 0.05)#120) #15
	parser.add_argument('--batch_size', type=int, default=1000, help='Training batch size')
	parser.add_argument('--noise_term', type=float, default=0.1, help='Noise factors')
	parser.add_argument('--epoch', type=int, default=20, help='Training epcohs')
	args = parser.parse_args()
	print(args)

	if args.train: 
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="0"

		import noisy
		'''
		model =  noisy.syn_noise(args)
		acc, mse = model.train()
		'''
		para = [] 
		count = 0.1 
		for i in range(17):
			para.append(np.round(count, 2))
			count += 0.05 
		#para =[0.9]
		acc_list = []
		mse_list = []
		noise_term_list = []
		for i in para: 
			print("***************************************")
			args.noise_term = i
			print(args)
			tf.reset_default_graph()
			model =  noisy.syn_noise(args)
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
		final.to_csv("noise_new.csv", index=False)
		
	elif args.train_1:
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="1"
		import noisy
		
		model =  noisy.syn_noise(args)
		acc, mse = model.train()
		
		para = [] 

		count = 0.1 
		for i in range(17):
			para.append(count)
			count += 0.05 
		
		#para = [0.1, 0.9]
		acc_list = []
		mse_list = []
		noise_term_list = []
		for i in para: 
			print("***************************************")
			args.noise_term = i
			print(args)
			tf.reset_default_graph()
			model =  noisy.syn_noise(args)
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
		final.to_csv("noise_new.csv", index=False)
		

	else:
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="1"
		import noisy
		model = noisy.syn_noise(args)
		model.train()