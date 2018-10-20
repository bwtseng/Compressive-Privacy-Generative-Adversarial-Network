# Compressive Privacy Generative Adversarial Networks

Machine learning has become more and more important in our life recent years and has also been applied to many cloud application. If a local user wants to use the face recognition system, she just needs to upload her private facial image to the cloud, and cloud server will give the recognition results that she is Alice. However, this raises a problem that the adversary in the cloud can intrude Alice's image since there is no privacy preserving mechanism applied to this system. In order to solve this urgent issue, we propose the compressive privacy generative adversarial network which is a local encryption scheme to prevent the private data of each user from being exposed in the cloud. With our proposed CPGAN, cloud only receive the compressive data and use them to do other machine learning task; we confirm that our CPGAN can get the quite accuracy compared with the state-of-the-art method over recent three years even use these compressive data, and it also persists certain reconstruction attack launched by malicious attacker. In short, our proposed CPGAN aims to minimize the trade off between privacy and utility, in other words, our target is to find the optimal neural network with hight utility gain but low privacy cost. 

## Dataset

* [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) - Avaible on the Internet
* [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) - Avaible on the Internet
* [SVHN](http://ufldl.stanford.edu/housenumbers/) - Avaible on the Internet

### Prerequisites
Our code is not limited to the version of thers packages bsides the tensoflow. We just

list the version of the packages installed in our server as follow: 
```
Windows 10/Ubuntu 16.0.5 
Tensorflow == 1.6.0 
Keras == 2.0.5
Scikit-Learn == 0.20
Numpy == 1.15
Scipy == 1.15
Matplotlib == 1.15
```

### Execution for training/testing (Utility)

Command for training/testing CPGAN. Nota the excuting testing command only get the utility accuracy. 

#Cifar10
```
Ubuntu : bash cifar10_run.sh 
Python L python main_cifar10.py --train True 
```

#SVHN
```
Ubuntu : bash svhn_run.sh $1 $2 $3
Python L python main_wrs.py --train True 
```

#Single CelebA

$1 : path to the images of the dataset
$2 : path to the lables of the dataset
$2 : True (If training phase)
$3 : False (If testing phase)
```
Ubuntu : bash single_celeba_run.sh $1 $2 $3 $4
Windows : python L python main_celeba.py --train True 
```

#Multi CelebA
$1 : path to the images of the dataset
$2 : path to the lables of the dataset
$2 : True (If training phase)
$3 : False (If testing phase)
```
Ubuntu : bash multi_celeba_run.sh $1 $2 $3 $4
Windows : Python L python main_muti.py --train True 
```

### Execution for training/testing (Privacy)

Excuting command of the script in different experiments for both training and testing:

#Cifar10
```
Ubuntu : bash cifar10_reco_run.sh 
Python L python main_cifar10_reco.py --train True 
```

#SVHN
```
Ubuntu : bash svhn_reco_run.sh $1 $2 $3
Python L python main_wrs_reco.py --train True 
```

#Single CelebA

$1 : path to the images of the dataset
$2 : path to the lables of the dataset
$2 : True (If training phase)
$3 : False (If testing phase)
```
Ubuntu : bash single_celeba_reco_run.sh $1 $2 $3 $4
Windows : python L python main_reco.py --train True 
```

#Multi CelebA
$1 : path to the images of the dataset
$2 : path to the lables of the dataset
$2 : True (If training phase)
$3 : False (If testing phase)
```
Ubuntu : bash multi_celeba_reco_run.sh $1 $2 $3 $4
Windows : Python L python main_multi_reco.py --train True 
```


### Experiments for the optimal Reconstructor 

Explain what these tests test and why

```
Give an example
```

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.


## Authors

* **Bo-wei Tseng** - *NTU* - [PurpleBooth](https://github.com/R06942098)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* My advisor : Pei-yuan Wu (Assistant Professor, NTU)
* Facenet : https://github.com/davidsandberg/facenet
* Compressive Privacy: From Information\/Estimation Theory to Machine Learning (Prof.Kung)
