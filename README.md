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

LGBM:
https://www.kaggle.com/anushkini/taxi-lightgbm

A brief description of the files and folders:<br>
-- TeamAPP_FinalReport.pdf - The final report of our project.<br>
-- demo.py - A demo script which shows our recommender system in action.<br>
--feature_engineering.py - A script which describes the feature engineering performed on the data.<br>
--final_pipeline.py - The final pipeline code for our project.<br>
--model.txt - The final LGBM model with an RMSE of 2.93.<br>
--test_df.feather - A feather file which contains the test dataset in compressed form.<br>
--visualization.py - A python script of all the visualizations performed on the dataset.<br>
--Models:<br>
----kNN Model/knn.py - Script to train the K Nearest Neighbours Model.<br>
----ANN/ann.py - Script to train the Neural Network Model.<br>
----LGBM/lgbm.py - Script to train the LGBM Model.<br>
----XGBoost/XGBoost.py - Script to train the XGBoost Model.<br>


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
</table>
