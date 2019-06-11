# Compressive Privacy Generative Adversarial Networks
Thie repo includes all ihe implementation of every experiments in our paper, and it's mainly based on python and tensorflow 
## **Abstract**
Machine learning as a service (MLaaS) has brought much convenience to our daily lives recently. However, the fact that the service is provided through cloud raises privacy leakage issues. In this work we propose the compressive privacy generative adversarial network (CPGAN), a data-driven adversarial learning framework for generating compressing representations that retain utility comparable to state-of-the-art, with the additional feature of defending against reconstruction attack. This is achieved by applying adversarial learning scheme to the design of compression network (privatizer), whose utility/privacy performances are evaluated by the utility classifier and the adversary reconstructor, respectively. Experimental results demonstrate that CPGAN achieves better utility/privacy trade-off in comparison with the previous work, and is applicable to real-world large datasets.
## **Model**

- CPGAN infrastructure
![image](https://github.com/R06942098/CPGAN/blob/master/cpgan_fig.png)

- Multiple adversaries strategy
![image](https://github.com/R06942098/CPGAN/blob/master/mul_adv.png)


## **Prerequisites**
We list the version of the packages on our PC. You can use the latest tensorflow and ubuntu version to run all these code. Note that the GPU we use is Tsela P100.
```
Windows 10/Ubuntu 16.0.5  
Tensorflow == 1.6.0 
Keras == 2.1.5 
Scikit-Learn == 0.20
Numpy == 1.14.3
Scipy == 1.1.0
Matplotlib == 2.2.2
```

## **Submitted Paper**
[Compressive Privacy Generative Adversarial Networks](https://drive.google.com/file/d/1KJiNZ9y59r3HLvsKfTqU1Oe85zGlFiMo/view?usp=sharing) 

## **Authors**

* **Bo-Wei Tseng** - *NTU* - [Github](https://github.com/R06942098)

* **Prof. Pei-Yuan Wu** - *NTU*- 




