# DA project repository for teamAPP

Links to various kernels:
Data feature engineering:<br>
https://www.kaggle.com/pradyu99914/data-feature-engineering<br>
<a href = "https://www.kaggle.com/pradyu99914/data-feature-engineering?scriptVersionId=21783669">Version 29</a> has supporting visualizations, while version 32 adds more features to the existing dataset.
Please note that the code has been executed over a number of commits.(as it takes ~7H per feature)

Linear, Ridge, Lasso regression, and Random forest model with 100 estimators:
https://www.kaggle.com/anushkini/nyc-taxi-fare-models 


Data Visualizations (visualizations that gave insights into feature engineering new columns):
https://www.kaggle.com/anushkini/nyc-taxi-fare-graphs

Xgboost: 
https://www.kaggle.com/anushkini/taxi-xgboost

kNN regression with vizualisation for best k value:
https://www.kaggle.com/pradyu99914/nyc-taxi-fare-models

Results:<br>
We have been able to obtain an RMSE rate of about 2.93 on the kaggle competition.<br>

Kaggle Submissions(RMSE error rate on the test set):<br>
<a href = "https://drive.google.com/file/d/1yECttTfHGwCP7I-x2DqwKL2goEplKj4r/view?usp=sharing">Link - 1(Xgboost)</a><br>
<a href = "https://drive.google.com/file/d/1Z90VlckqNv_5gVYRxhFm5A1L9x89k3wz/view?usp=sharing">Link - 2(kNN, regression, random forest)(description for each submission provided)</a>

All kaggle submissions made till date:
<table class="tg">
  <tr>
    <th class="tg-yw4l"><b>Model</b></th>
    <th class="tg-yw4l"><b>Model details</b></th>
    <th class="tg-yw4l"><b>RMSE</b></th>
  </tr>
  <tr>
    <td class="tg-yw4l">XGBoost</td>
    <td class="tg-yw4l">Trained on 1 Million data points</td>
    <td class="tg-yw4l">4.46939</td>
  </tr>
  <tr>
    <td class="tg-yw4l">XGBoost (Bagging)</td>
    <td class="tg-yw4l">Trained on 8 Million data points</td>
    <td class="tg-yw4l">4.19958</td>  
  </tr>
  <tr>
    <td class="tg-yw4l">XGBoost (Bagging)</td>
    <td class="tg-yw4l">Trained on 54 Million data points</td>
    <td class="tg-yw4l">4.12760</td>  
  </tr>
  <tr>
    <td class="tg-yw4l">XGBoost</td>
    <td class="tg-yw4l">Used Bayeisan Optimization for hyperparameter tuning</td>
    <td class="tg-yw4l">4.18783</td>  
  </tr>
  <tr>
    <td class="tg-yw4l">XGBoost (Bagging)</td>
    <td class="tg-yw4l">Improved Dataset with feature Engineering, 8 Million data points</td>
    <td class="tg-yw4l">3.91798</td>  
  </tr>
  <tr>
    <td class="tg-yw4l">XGBoost</td>
    <td class="tg-yw4l">Improved dataset and used Bayesian Optimization</td>
    <td class="tg-yw4l">3.17963</td>  
  </tr>  
  <tr>
    <td class="tg-yw4l">XGBoost</td>
    <td class="tg-yw4l">Converted Coordinates from decimals into radians and Bayesian Optimization</td>
    <td class="tg-yw4l">3.17697</td>  
  </tr>
    <tr>
    <td class="tg-yw4l">XGBoost</td>
    <td class="tg-yw4l">Feature Engineered Day, Month columns to the dataset</td>
    <td class="tg-yw4l">3.11282</td>  
  </tr>
  <tr>
    <td class="tg-yw4l">LGBM</td>
    <td class="tg-yw4l">Improved Dataset</td>
    <td class="tg-yw4l">3.13226</td>  
  </tr>
  <tr>
    <td class="tg-yw4l">LGBM</td>
    <td class="tg-yw4l">Changed boosting hyperparameters</td>
    <td class="tg-yw4l">3.08951</td>  
  </tr>
  <tr>
    <td class="tg-yw4l">LGBM</td>
    <td class="tg-yw4l">Converted Coordinates from decimals into radians</td>
    <td class="tg-yw4l">3.08830</td>  
  </tr>
  <tr>
    <td class="tg-yw4l">LGBM</td>
    <td class="tg-yw4l">Feature Engineered Day, Month columns to the dataset</td>
    <td class="tg-yw4l">3.02238</td>  
  </tr>
  <tr>
    <td class="tg-yw4l">LGBM</td>
    <td class="tg-yw4l">Reworked Dataset with more Feature Engineering</td>
    <td class="tg-yw4l">2.99228</td>  
  </tr>
  <tr>
    <td class="tg-yw4l">LGBM</td>
    <td class="tg-yw4l">Added new distance features</td>
    <td class="tg-yw4l">2.99095</td>  
  </tr>
  <tr>
    <td class="tg-yw4l">LGBM</td>
    <td class="tg-yw4l">Trained on 15 Million data points</td>
    <td class="tg-yw4l">2.93553</td>  
  </tr>
  <tr>
    <td class="tg-yw4l">Linear Regression</td>
    <td class="tg-yw4l">Baseline Model</td>
    <td class="tg-yw4l">5.39</td>  
  </tr>
  <tr>
    <td class="tg-yw4l">Linear Regression</td>
    <td class="tg-yw4l">Final model trained on engineered data</td>
    <td class="tg-yw4l">5.18</td>  
  </tr>
  <tr>
    <td class="tg-yw4l">Ridge ression</td>
    <td class="tg-yw4l">Ridge regression with grid search</td>
    <td class="tg-yw4l">5.18</td>  
  </tr>
 
  <tr>
    <td class="tg-yw4l">Lasso regression</td>
    <td class="tg-yw4l">Lasso regression</td>
    <td class="tg-yw4l">9.409</td>  
  </tr>
  <tr>
    <td class="tg-yw4l">Lasso regression</td>
    <td class="tg-yw4l">Lasso regression with grid search</td>
    <td class="tg-yw4l">5.05</td>  
  </tr>
  <tr>
    <td class="tg-yw4l">Random forest regressor</td>
    <td class="tg-yw4l">Random forest regressor</td>
    <td class="tg-yw4l">4.43</td>  
  </tr>
  <tr>
    <td class="tg-yw4l">Linear Regression</td>
    <td class="tg-yw4l">Final model trained on engineered data</td>
    <td class="tg-yw4l">5.18</td>  
  </tr>
  <tr>
    <td class="tg-yw4l">kNN regressor with bagging</td>
    <td class="tg-yw4l">using feature engineered data</td>
    <td class="tg-yw4l">3.54</td>  
  </tr>
  <tr>
    <td class="tg-yw4l">Artificial neural network</td>
    <td class="tg-yw4l">ANN with normalization of the data</td>
    <td class="tg-yw4l">5.18</td>  
  </tr>
</table>
