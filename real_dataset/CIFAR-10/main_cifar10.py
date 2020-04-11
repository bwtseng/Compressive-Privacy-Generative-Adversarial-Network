import argparse
import os 

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='CIFAR-10')
	parser.add_argument('--train_0', type=bool, default=False, help='Training with gpu:0')
	parser.add_argument('--train_1', type=bool, default=False, help='Training with gpu:1')
	parser.add_argument('--test', type=bool, default=False, help='Inference')
	parser.add_argument('--com_dim', type=int, default=1024, help="Compressive Dimension")
	parser.add_argument('--model_path',type=str,default='Path to the model file',help='Model Path')
	args = parser.parse_args()			

	
	if args.train_0: 

		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="0"
		import cifar10
		model = cifar10.shakenet(args)
		model.train()

	elif args.train_1 :

		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="1"
		import cifar10
		model = cifar10.shakenet(args)
		model.train() 
	else:
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="0"
		import cifar10
		model = cifar10.shakenet(args)
		model.test()
	
