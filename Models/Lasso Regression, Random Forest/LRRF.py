#package imports
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn import linear_model
import numpy as np

#read the test and train sets
train_df= pd.read_feather('../input/kernel318ff03a29/nyc_taxi_data_raw.feather')
test_df = pd.read_feather('../input/kernel318ff03a29/test_feature.feather')

#select all rows, and all columns after the second column
X = train_df.iloc[:,3:]
#target variable
y = train_df['fare_amount']
#select all rows, and all columns after the second column
X_test = test_df.iloc[:,2:]
#reorder the columns
X_test = X_test[X.columns]
X.head()
X_test.head()
import gc
gc.collect()

# fit a normal liner regression model to this dataset.
#Note: this library uses the closed form expression for the parameters and not gradient descent
model = linear_model.LinearRegression()
model.fit(X,y)

model = linear_model.Lasso(normalize = True)
gc.collect()
model.fit(X,y)

print("R2 value of model",model.score(X,y))
y_test = model.predict(X_test)
holdout = pd.DataFrame({'key': test_df.key, 'fare_amount': y_test})
#write the submission file to output
holdout.to_csv('submission_lasso.csv', index=False)

#random forest regressor
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(max_depth=4, random_state=0, n_estimators=100)
gc.collect()
model.fit(X,y)
y_test = model.predict(X_test)
holdout = pd.DataFrame({'key': test_df.key, 'fare_amount': y_test})
#write the submission file to output
holdout.to_csv('submission_rf.csv', index=False)
