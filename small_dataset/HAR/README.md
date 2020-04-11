# HAR Dataset
## **Data Information**

- Pleaser refer to the [link](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
- We proceed with the sequential (time-series) format. Use the har.py to load the data.

## **Experimental Detail**
- Adopt classification accuracy as the utility evaluation metric.
- Privacy is evaluated by the adversaries (LRR, KRR, NN) 
achieving minimum mean square error.

## **Empirical Results**
- Trade-off between privacy and utility
![image](https://github.com/R06942098/CPGAN/blob/master/HAR/img/HAR_data_final_nonlinear.png)

- Neural networks not guarantee to attain minimum MSE as experimented in the DNN method.
![image](https://github.com/R06942098/CPGAN/blob/master/HAR/img/mse_comparison_har_dnn.png)

- Neural networks not guarantee to attain minimum MSE as experimented in the DNN(Resize) method.
![image](https://github.com/R06942098/CPGAN/blob/master/HAR/img/mse_comparison_har_pca.png)


## **Excecution**
```
bash compile.sh
```
Note that you can tune different parameters defined in the "main" file.
