import argparse
import os 
import tensorflow as tf 
import pandas as pd 

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Attack')
	parser.add_argument('--train',type=bool,default=False,help='Training')
	parser.add_argument('--ori_dim', type=int, default=32, help='Orignal dimension')
	parser.add_argument('--com_dim', type=int, default=16, help='Compressive dimension')
	parser.add_argument('--prior_prob', type=float, default=0.5, help='Prior probability')
	parser.add_argument('--batch_size', type=int, default=1000, help='Training batch size')
	parser.add_argument('--samples', type=int, default=22000, help='Training data size')
	parser.add_argument('--trade_off', type=float, default = 1, help='Trade off term')
	parser.add_argument('--noise_term', type=float, default=0.0001, help='Noise factorsss')
	parser.add_argument('--epoch', type=int, default=200, help='Training epcohs')
	parser.add_argument('--seed', type=int, default=9)
	parser.add_argument('--gamma', type=float, default=0.001)
	parser.add_argument('--mapping_dim',type=int, default=32)
	args = parser.parse_args()
	print(args)

	if args.train: 
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="0"
		import gd_opt
		
		#model = gd_opt.gd(args)
		#model.train()

		
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
		theory_acc_list = []
		theory_mse_list = []
		lamda_list = []
		loops = 1
		for i in para: 

			print("***************************************")
			args.trade_off = i
			print(args)
			tf.reset_default_graph()
			model = gd_opt.gd(args)
			theory_acc, theory_mse, acc, mse = model.train()
			acc_list.append(acc)
			mse_list.append(mse)
			theory_acc_list.append(theory_acc)
			theory_mse_list.append(theory_mse)
			lamda_list.append(i)
			print("***************************************")

			if loops % 10 == 0 :
				Matrix = {}
				Matrix["Lambda"] = lamda_list
				Matrix['acc']= acc_list
				Matrix['acc_theory'] = theory_acc_list
				Matrix['mse'] = mse_list
				Matrix['mse_theory'] = theory_mse_list
				final = pd.DataFrame(Matrix)
				final.to_csv("Final_theory.csv", index=False)
			loops+=1

		Matrix = {}
		Matrix["Lambda"] = lamda_list
		Matrix['acc']= acc_list
		Matrix['acc_theory'] = theory_acc_list
		Matrix['mse'] = mse_list
		Matrix['mse_theory'] = theory_mse_list
		final = pd.DataFrame(Matrix)
		final.to_csv("Final_theory.csv", index=False)
				


	elif args.train_1 :
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="1"
		import gd_opt
		model = gd_opt.gd(args)
		model.train()


	else:
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="1"
		import gd_opt
		model = gd_opt.gd(args)
		model.train()