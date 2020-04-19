# GENKI-4K Dataset
## Data Information
- Consist of 4000 images, 2800 for training while 1200 for testing.
- Detect whether the facial image is smile or not.

## Experimental Detail

- Adopt classification accuracy as the utility evaluation metric.
- Privacy is evaluated by the adversaries (LRR, KRR, NN) 
achieving minimum mean square error.
## Empirical Results
- Trade-off between privacy and utility
<center> <img src="img/Genki4K_data_final_nonlinear.png" width="400" height="250"> </center>

- Reconstructed images from five privacy preserving methods.
<center> <img src="img/img1.png.png" width="400" height="250"> </center>

- Neural networks not guarantee to attain minimum MSE.
<center> <img src="img/mse_comparison_Genki_cpgan.png" width="400" height="250"> </center>

## Excecution
```
bash compile.sh
```
Note that you can tune different parameters defined in the "main" file.
