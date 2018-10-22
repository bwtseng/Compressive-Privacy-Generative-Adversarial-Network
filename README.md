# Compressive Privacy Generative Adversarial Networks
## **Abstract**
Machine learning has become more and more important in our life recent years and has also been applied to many cloud application. If a local user wants to use the face recognition system, she just needs to upload her private facial image to the cloud, and cloud server will give the recognition results that she is Alice. However, this raises a problem that the adversary in the cloud can intrude Alice's image since there is no privacy preserving mechanism applied to this system. In order to solve this urgent issue, we propose the compressive privacy generative adversarial network which is a local encryption scheme to prevent the private data of each user from being exposed in the cloud. With our proposed CPGAN, cloud only receive the compressive data and use them to do other machine learning task; we confirm that our CPGAN can get the quite accuracy compared with the state-of-the-art method over recent three years even use these compressive data, and it also persists certain reconstruction attack launched by malicious attacker. In short, our proposed CPGAN aims to minimize the trade off between privacy and utility, in other words, our target is to find the optimal neural network with hight utility gain but low privacy cost. 

Including all ihe implentation python/bash script and reference in this CPGAN reco.

## **Dataset**

* [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) - Avaible on the Internet
* [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) - Avaible on the Internet
* [SVHN](http://ufldl.stanford.edu/housenumbers/) - Avaible on the Internet

## **Prerequisites**
Our code is not limited to the version of these packages besides tensoflow. We just

list the version of the packages installed in our server as follow: 
```
Windows 10/Ubuntu 16.0.5 
Tensorflow == 1.6.0 
Keras == 2.1.5
Scikit-Learn == 0.20
Numpy == 1.14.3
Scipy == 1.1.0
Matplotlib == 2.2.2
```

## **Preprocessing of CelebA dataset**
Thank [FaceNet](https://github.com/davidsandberg/facenet) for the best MTCNN aligned open source code.
```bash
├── Dataset
    ├── CelebA
    	├── 000000.png
    	├── 000001.png
    	├── XXXXXX.png
    	└── 202599.png
```

```
python align_dataset_mtcnn.py input_dir output_dir --image_size 112 --margin 32 --random_order
```


## **Execution for training/testing (Utility)**

Command for training/testing CPGAN. Nota the excuting testing command only get the utility accuracy. 

### **Cifar10**
Load data with the built-in package from Keras.
```
At first, make a directory cifar10_log (e.g mkdir cifar10_log in ubuntu)
Ubuntu : bash cifar10_run.sh $1 $2 $3 $4
Windows : python main_cifar10.py --train True (Training)

Ubuntu : bash cifar10_run.sh $1 $2 $3 $4
Windows: python main_cifar10.py --test True --model_path your_path/to/cifar_log/model_checkpoint(Training)
```

### **SVHN**
Please build the directory of the SVHN as follow:
```bash
├── svhn
    ├── train_32x32.mat
    ├── test_32x32.mat
    └── extra_32x32.mat
```

```
At first, make a directory svhn_log (e.g mkdir svhn_log in ubuntu)
Windows/Ubuntu : python main_wrs.py --train True (Training)
Windows/Ubuntu : python main_wrs.py --test True  (Testing)
```

### **Single CelebA**
If you not use the preprocessing from our Github, please build the CelebA directory as follow : 
```bash
├── dataset
    ├── 000001.png
    ├── 000002.png
    ├── XXXXXX.png  
    └── 202599.png
```

```
At first, make a directory single_celebA (e.g mkdir single_celebA in ubuntu)
Windows/Ubuntu : python main_celeba.py --train True  --attritube Male --path dataset_path 
Windows/Ubuntu : python main_celeba.py --test True --model_path your_path
```
### **Multi CelebA**

```
At first, make a directory multicpgan_log (e.g mkdir multicpgan_log in ubuntu)
Windows/Ubuntu : python main_mutli.py --train True --path dataset_path (training)
Windows/Ubuntu : python main_multi.py --test True --path dataset_path --model_path your_path (testing)
```

## **Execution for training/testing (Privacy)**

Excuting command of the script in different experiments for both training and testing:

### **Cifar10**
Load data with the built-in package from Keras.
```
At first, make a directory cpgan_log (e.g mkdir cpgan_log in ubuntu)
Windows/Ubuntu : python main_cifar10_reco.py --train True (Training)
Windows/Ubuntu : python main_cifar10_reco.py --test True (Testing, only plot the training losss curve)
```

### **SVHN**
Please build the directory of the SVHN as follow:
```bash
├── svhn
    ├── train_32x32.mat
    ├── test_32x32.mat
    └── extra_32x32.mat
```


```
Windows/Ubuntu : python main_wrs_reco.py --train True (Training)
Windows/Ubuntu : python main_wrs_reco.py --test True (Testing, only plot the training losss curve )
```

### **Single CelebA**
If you not use the preprocessing from our Github, please build the CelebA directory as follow : 
```bash
├── dataset
    ├── 000001.png
    ├── 000002.png
    ├── XXXXXX.png 
    └── 202599.png
```

```
At first, make a directory single_celebA (e.g mkdir single_celebA in ubuntu)
Windows/Ubuntu : python main_reco.py --train True --path dataset_path (Training)
Windows/Ubuntu : python main_reco.py --test True --path dataset_path --model_path model_path (Testing) 
```

### **Multi CelebA**

```
At first, make a directory multicpgan_log (e.g mkdir multicpgan_log in ubuntu)
Windows/Ubuntu : python main_multi_reco.py --train True --path dataset_path (Training)
Windows/Ubuntu : python main_multi_reco.py --test True  --path dataset_path --model_path model_path (Testing)
```


## **Experiments for the optimal Reconstructor**

Explain what these tests test and why

```
Give an example
```

## **Original Paper**

[Compressive Privacy Generative Adversarial Networks](https://drive.google.com/file/d/1UOFbkf4Eq9TfF1_edXAkIVOVnvbVLY6Q/view?usp=sharing) 

## **Authors**

* **Bo-wei Tseng** - *NTU* - [PurpleBooth](https://github.com/R06942098)


## **License**

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## **Acknowledgments**

* My advisor : Pei-yuan Wu (Assistant Professor, NTU)
* Facenet : https://github.com/davidsandberg/facenet
* Compressive Privacy: From Information\/Estimation Theory to Machine Learning (Prof.Kung)
