# MNIST Dataset
## **Data Information**

- Pleaser refer to the [link](http://yann.lecun.com/exdb/mnist/)
- We use the package from tensorflow to load whole data.

## **Experimental Detail**
- Adopt classification accuracy as the utility evaluation metric.
- Privacy is evaluated by the adversaries (LRR, KRR, NN) 
achieving minimum mean square error.

## **Empirical Results**
- Trade-off between privacy and utility
![image](https://github.com/R06942098/CPGAN/blob/master/MNIST/img/MNIST_data_final_nonlinear.png)

## **Excecution**
```
bash compile.sh
```
Note that you can tune different parameters defined in the "main" file.
