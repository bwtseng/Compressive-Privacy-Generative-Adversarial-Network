import argparse
import os 

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Multi adversary CPGAN')
	#parser.add_argument('--train_0', type=bool, default=False, help='Training with gpu:0')
	#parser.add_argument('--train_1', type=bool, default=False, help='Training with gpu:1')
    parser.add_argument('--train', default=False, action='store_true', help='Training phase' )
    parser.add_argument('--test',  default=False, action='store_true', help='Testing phase')
    parser.add_argument('--data_dir',type=str,default='/path/to/your/celeba_dataset', help='Path to your facial images.')
    parser.add_argument('--label_dir', type=str, default='/path/to/your/label_file', help='Path to your label csv file.')
    parser.add_argument('--model_dir', type=str, default='/path/to/your/trained_model', help='Pre-trained model path.')
	parser.add_argument('--name', type=str, default='multi-task-celeba')
    parser.add_argument('--attribute', '-a', type=str, default='Male', help='Which attributes you specify')
    parser.add_argument('--com_dim', type=int, default=2, help='Compressive dimension')
	parser.add_argument('--noise', default=False, action='store_true', help='Whether adding noise or not')
	parser.add_argument('--batch_size', '-b', type=int, default=1024, help='Batch_size')
	parser.add_argument('--citer', type=int, default=25, help='Adversary iteration')
	parser.add_argument('--gamma', type=float, default=1, help='Variance of the kernel function')
	parser.add_argument('--seed', type=int, default=1, help='The sampled weights of the RFF mapping')
	parser.add_argument('--mapping_dim', type=int, default=5000, help='Dimension of the intrinsic space')
	
	args = parser.parse_args()
	print(args)
	
    if args.train:
        import multi_celeba as mc 
        model = mc.cpgan(args)
        model.train()
    elif args.test: 
        import multi_celeba as mc  
        model = mc.cpgan(args)
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