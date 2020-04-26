# GENKI-4K Dataset
## GENKI-4K Dataset Info.
- Please refer to the [link](http://mplab.ucsd.edu) 
- This dataset is consisted of 4000 images, including 2800 training examples and 1200 testing examples, respectively. Each data sample is along with experession labels and pose labels. In our experiment, only experssion label is used. 

## Evaluation Metrics
- Adopt classification accuracy as the utility evaluation metric.
- Privacy is evaluated by the reconstruction error between original and reconstructed images, we use l2 norm here and it may have other alternative in the near future.

Our CPGAN will be compared with four different mechanisms, including DNN, DNN (Resize), Noist (act as differential privacy) and RAN. More detail about these comparison are elaborated in our manuscript Sevtion V 

## Empirical Results
- Trade-off between privacy and utility
    <center> <img src="img/Genki4K_data_final_nonlinear.png" width="400" height="250"> </center>

- Reconstructed images from five privacy preserving methods.
    <center> <img src="img/img1.png" width="400" height="250"> </center>

- Neural network adversary is not always the most intrusive, confirmed by the figure below. It also proves that our multiple adversary strategy is useful in some cases.
    <center> <img src="img/mse_comparison_Genki_cpgan.png" width="400" height="250"> </center>

We also provide the experimental records for these mechanisms, which are csv format file under the numberical-file folder. Each file records the MSE and accuracy with different trade-off parameters. Regarding to this paramters, please refer to our manuscript for more detail.
## Execution
```
python main_dnn.py --train True 
python main_hybrid.py --train True 
python main_ran.py --train True 
python main_noise.py --train True 
```
The rest of the argument are not listed here, if you want to try different parameters, please refer to our released code, in which the file name start with "main".
