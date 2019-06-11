import argparse
import os 

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Attack')
	parser.add_argument('--train_0', type=bool, default=False, help='Training with gpu:0')
	parser.add_argument('--train_1', type=bool, default=False, help='Training with gpu:1')
	parser.add_argument('--test', type=bool, default=False, help='Testing')
	parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
	parser.add_argument('--cut_out', type=bool, default=False, help='cut_out')
	parser.add_argument('--path',type=str,default='/path/to/mat/file', help='Path to your dataset folder')
	parser.add_argument('--gamma', type=float, default=1, help='Variance of the kernel function')
	parser.add_argument('--seed', type=int, default=1, help='The sampled weights of the RFF mapping')
	parser.add_argument('--mapping_dim', type=int, default=5000, help='Dimension of the intrinsic space')
	args = parser.parse_args()
	print(args)

	if args.train_0: 
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="0"
		import svhn_multi_r 
		model = svhn_multi_r.wrs(args)
		model.train()

	elif args.train_1 :
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="1"
		import svhn_multi_r 
		model = svhn_multi_r.wrs(args)
		model.train() 


	else:
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="0"
		import svhn_multi_r 
		model = svhn_multi_r.wrs(args)
		model.test()