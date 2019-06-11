# HAR Dataset
## **Data Information**

- Sample 20k datae from two gaussian mixture model (for label 0/1) which has same covariance matrix.
- Guarantee our CPGAN trained by gradient descent achieves accuracy apporximate to theoretical results.

## **Experimental Detail**
- Adopt classification accuracy as the utility evaluation metric.
- Privacy is evaluated by the adversaries (LRR, KRR, NN) 
achieving minimum mean square error.
- There are some mathematical formula are listed in the paper, you can refer the pdf file for more information.

## **Empirical Results**
- Trade-off between privacy and utility
![image](https://github.com/R06942098/CPGAN/blob/master/HAR/img/HAR_data_final_nonlinear.png)

- Our CPGAN achieves the results similar to theorectical analysis.
![image](https://github.com/R06942098/CPGAN/blob/master/HAR/img/mse_comparison_har_dnn.png)


## **Excecution**
```
bash compile.sh
```
Note that you can tune different parameters defined in the "main" file.
