# SVHN Dataset

## Data Information

- Please refer to the [link](http://ufldl.stanford.edu/housenumbers/)

- Download the data, plear follow the command line: 

  ```
  bash down.sh
  ```

## Evaluation Metrics

- Adopt classification accuracy as sthe utility evaluation metric.
- Privacy is evaluated by qualitative results, that is, we evaluate the quality of the reconstructed images under human perception, as there is no fair comparison in the literature using the mean square loss.

## Experimental Detail

- Adopt classification accuracy as the utility evaluation metric.
- Privacy is evaluated by qualitative results (i.e. reconstructed images).

## **Empirical Results**

- Reconstructions:

  <center> <img src="img/svhn_fig_res.png" width="450" height="200"></center>	

  The first row of each figure consists of randomly sampled original images.  The second row consists of the reconstructed images assuming the adversary acquires the original image.  The last row consists of the images reconstructed from the compressing representations under white-box attack:

- Utility Accuracy:

  | Model     | Accuracy (%) |
  | :-:       | :-:      |
  | ResNet-20 | 97.70   |
  | Zagoruyko | 98.46    |
  | Xavier    | 98.6   |
  | Zagoruyko | 97.68   |

# Execution 

```
python main_cifar_cpgan.py --train True --data_dir "Your Path" --label_dir "Your path""
```

The rest of the argument are not listed here, if you want to tune the parameters, please refer to the the file, which start with "main".

