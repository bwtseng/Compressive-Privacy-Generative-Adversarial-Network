# HAR Dataset
## HAR Dataset Info.
- Pleaser refer to the [link](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
- Our experiments use sequential (time-series) format of this dataset. You can use our data loader function (har.py) to get the data in this format without any efforts.   

## Evaluation Metrics
- Adopt classification accuracy as the utility evaluation metric.
- Privacy is evaluated by the reconstruction error between original and reconstructed images, and the adversary (LRR, KRR, NN) achieving the minimum MSE is chosen. Note that we use l2 norm here and it may have other alternative in the near future.

## Empirical Results
- Trade-off between privacy and utility
    <center> <img src="img/HAR_data_final_nonlinear.png" width="400" height="250"> </center>

- Neural network is not always the most intrusive adversary, confirmed by the following two figures. Note that top figure is DNN method and bottom one is DNN (Resize) method, respectively.
    <center> <img src="img/mse_comparison_har_dnn.png" width="400" height="250"> </center>
    <center> <img src="img/mse_comparison_har_pca.png" width="400" height="250"> </center>
We also provide the experimental records for these mechanisms, which are csv format file under the numberical-file folder. Each file records the MSE and accuracy with different trade-off parameters. Regarding to this paramters, please refer to our manuscript for more detail.

## Execution
```
python main_dnn.py --train True 
python main_hybrid.py --train True 
python main_ran.py --train True 
python main_noise.py --train True 
```
The rest of the argument are not listed here, if you want to try different parameters, please refer to our released code, in which the file name start with "main".
