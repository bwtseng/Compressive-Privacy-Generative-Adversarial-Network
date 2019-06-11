# CelebA Dataset
## **Data Information**
- Please refer to the [link](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- Use facenet source (refer to the folder "understand facenet") code to align and crop the images.

## **Experimental Detail**
- Adopt classification accuracy as the utility evaluation metric.
- Privacy is evaluated by qualitative results (i.e. reconstructed images).

## **Empirical Results**

### Reconstructed images from single task CPGAN.
- The first row consists of the original images sampled from CIFAR-10 dataset, second row consists of the reconstructed images assuming that privatizer is identity function, and the last row consists of the images reconstructed images from the compressing representations.

![image](https://github.com/R06942098/CPGAN/blob/master/CelebA/img/single_celeba_res.png)

| Model     | Accuracy |
| ---       | ---      |
| LNets+ANets | 87.30% |
| Zhong | 89.97%   |
| CPGAN | 89.92%   |

### Reconstructed images from multi task CPGAN.
- The first row consists of the original images sampled from CIFAR-10 dataset, second row consists of the reconstructed images assuming that privatizer is identity function, and the last row consists of the images reconstructed images from the compressing representations.

![image](https://github.com/R06942098/CPGAN/blob/master/CelebA/img/multi_celeba_new.png)

| Model     | Accuracy |
| ---       | ---      |
| Han | 92.52%   |
| ATNET_GT | 90.18%    |
| CPGAN   | 90.30%   |

## **Excecution**
```
bash compile.sh
```
Note that you can tune different parameters defined in the "main" file.
