# CIFAR-10 Dataset
## **Data Information**
- Please refer to the [link](https://www.cs.toronto.edu/~kriz/cifar.html)
- We use the package from tensorflow to load the whole dataset.

## **Experimental Detail**
- Adopt classification accuracy as the utility evaluation metric.
- Privacy is evaluated by qualitative results (i.e. reconstructed images).

## **Empirical Results**

Reconstructed images
- The first row consists of the original images sampled from CIFAR-10 dataset, second row consists of the reconstructed images assuming that privatizer is identity function, and the last row consists of the images reconstructed images from the compressing representations.

![image](https://github.com/R06942098/CPGAN/blob/master/CIFAR-10/img/cifar_fig_res.png)

| Model     | Accuracy |
| ---       | ---      |
| ResNet-20 | 92.28%   |
| Zagoruyko | 96.2%    |
| Xavier    | 96.45%   |
| Zagoruyko | 93.87%   |

## **Excecution**
```
bash compile.sh
```
Note that you can tune different parameters defined in the "main" file.
