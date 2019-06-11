# CelebA Dataset
## **Data Information**
- Please refer to the [link](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- Use facenet source code to align and crop the images.

## **Experimental Detail**
- Adopt classification accuracy as the utility evaluation metric.
- Privacy is evaluated by qualitative results (i.e. reconstructed images).

## **Empirical Results**

### Reconstructed images from single task CPGAN.
- The first row consists of the original images sampled from CIFAR-10 dataset, second row consists of the reconstructed images assuming that privatizer is identity function, and the last row consists of the images reconstructed images from the compressing representations.

![image](https://github.com/R06942098/CPGAN/blob/master/SVHN/svhn_fig_res.png)

| Model     | Accuracy |
| ---       | ---      |
| ResNet-20 | 97.70%   |
| Zagoruyko | 98.46%    |
| Xavier    | 98.6%   |
| Zagoruyko | 97.68%   |

### Reconstructed images from multi task CPGAN.
- The first row consists of the original images sampled from CIFAR-10 dataset, second row consists of the reconstructed images assuming that privatizer is identity function, and the last row consists of the images reconstructed images from the compressing representations.

![image](https://github.com/R06942098/CPGAN/blob/master/SVHN/svhn_fig_res.png)

| Model     | Accuracy |
| ---       | ---      |
| ResNet-20 | 97.70%   |
| Zagoruyko | 98.46%    |
| Xavier    | 98.6%   |
| Zagoruyko | 97.68%   |

## **Excecution**
```
bash compile.sh
```
Note that you can tune different parameters defined in the "main" file.
