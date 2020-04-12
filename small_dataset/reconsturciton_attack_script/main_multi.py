import argparse
import os 

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Attack')
	parser.add_argument('--train',type=bool,default=False,help='Training')
	#parser.add_argument('--path',type=str,default='/home/pywu/bowei/crop_v',help='Path facial image')
	parser.add_argument('--path',type=str,default='/path/to/your/data/derectory',help='Path to facial image')
	parser.add_argument('--test',type=bool,default=False,help='Testing')
	parser.add_argument('--model_path',type=str,default='/path/to/your/model/path',help='Model path')

	#parser.add_argument('--train_1',type=bool,default=False,help='Training with gpu:1')
	args = parser.parse_args()
	print(args)

	if args.train : 
		import celeba_multi
		model = celeba_multi.cpgan(args)
		model.train()
	else : 
		import celeba_multi
		model = celeba_multi.cpgan(args)
		model.test()
	'''
	if args.train_0: 

		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="0"
		import celeba_multi
		model = celeba_multi.cpgan(args)
		model.train()

	elif args.train_1 :

		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="1"
		import celeba_multi
		model = celeba_multi.cpgan(args)
		model.train() 
	else:
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="1"
		import celeba_multi
		model = celeba_multi.cpgan(args)
		model.test()
	'''