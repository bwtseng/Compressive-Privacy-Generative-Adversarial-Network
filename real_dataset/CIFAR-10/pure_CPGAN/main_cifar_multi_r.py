import argparse
import os 

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='CIFAR-10 shakenet (multiple adversaries)')
	#parser.add_argument('--train_0', type=bool, default=False, help='Training with gpu:0')
	#parser.add_argument('--train_1', type=bool, default=False, help='Training with gpu:1')
	parser.add_argument('--train', default=False, action='store_true', help='training phase')
	parser.add_argument('--test', default=False, action='store_true', help='training phase')
	parser.add_argument('--gamma', type=float, default=1, help='Variance of the kernel function')
	parser.add_argument('--seed', type=int, default=1, help='The sampled weights of the RFF mapping')
	parser.add_argument('--mapping_dim', type=int, default=5000, help='Dimension of the intrinsic space')
	parser.add_argument('--com_dim', type=int, default=1024, help="Compressive Dimension")
	parser.add_argument('--batch_size', '-b', type=int, default=1024, help='Batch_size')
	parser.add_argument('--citer', type=int, default=25, help='Adversary iteration')
	parser.add_argument('--mapping_dim', type=int, default=5000, help='Dimension of the intrinsic space')
	parser.add_argument('--model_dir', type=str, default='/path/to/your/trained_model', help='Pre-trained model path.')
	parser.add_argument('--name', type=str, default='cifar10')
	parser.add_argument('--lr', type=int, default=0.2, help='Initial learning rate')
	parser.add_argument('--trade_off', type=int, default=10, help='Trade-off parameter between privacy and utility.')
	args = parser.parse_args()

    if args.train:
        import celeba_multi_r 
        model = celeba_multi_r.CPGAN(args)
        model.train()
    elif args.test: 
        import celeba_multi_r 
        model = celeba_multi_r.CPGAN(args)
        model.test()
    else:
    	raise ValueError("Plear input correct args !!!")


	"""
	if args.train_0: 
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="0"
		import celeba_multi_r 
		model = celeba_multi_r .cpgan(args)
		model.train()

	elif args.train_1 :
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="1"
		import celeba_multi_r 
		model = celeba_multi_r .cpgan(args)
		model.train() 


	else:
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="0"
		import celeba_multi_r 
		model = celeba_multi_r .cpgan(args)
		model.test()
	"""