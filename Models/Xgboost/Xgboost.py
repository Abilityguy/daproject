#Xgboost classifier

#package imports
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb
from bayes_opt import BayesianOptimization
import gc
from sklearn import linear_model
import numpy as np

#read the test and train sets
train_df= pd.read_feather('../input/kernel318ff03a29/nyc_taxi_data_raw.feather')
test_df = pd.read_feather('../input/kernel318ff03a29/test_feature.feather')

#examine the dataset's fist 5 rows
train_df.head()

#examine the test dataset
test_df.head()

#Number of training instances
len(train_df)

#for training the model on a part of the data. The whole dataset is too big to be trained with the present resources.
#Here, we are using a bagging model but the subsets of data are random samples without replacement
n_models = 8
chunk = 1_000_000
sample_data = []
for i in range(n_models):
    sample_data.append(train_df[i*chunk:(i+1)*chunk])
gc.collect()

Xs = []
ys = []
for i in range(len(sample_data)):
    #select all rows, and all columns after the second column
    X = sample_data[i].iloc[:,3:]
    #target variable
    y = sample_data[i]['fare_amount']
    #select all rows, and all columns after the second column
    Xs.append(X)
    ys.append(y)
    
del(sample_data)
X_test = test_df.iloc[:,2:]
#reorder the columns
X_test = X_test[X.columns]
gc.collect()

#The DMatrix is a data structure optimized for training an Xgboost classifier
dtrain = xgb.DMatrix(Xs[0], label=ys[0])
gc.collect()

#Reference: https://www.kaggle.com/btyuhas/bayesian-optimization-with-xgboost
def xgb_evaluate(max_depth, gamma, colsample_bytree):
    params = {'eval_metric': 'rmse',
              'max_depth': int(max_depth),
              'subsample': 0.8,
              'eta': 0.1,
              'gamma': gamma,
              'colsample_bytree': colsample_bytree}
    cv_result = xgb.cv(params,dtrain, num_boost_round=500, nfold=3)    
    
    # Bayesian optimization only knows how to maximize, not minimize, so return the negative RMSE
    return -1.0 * cv_result['test-rmse-mean'].iloc[-1]

xgb_bo = BayesianOptimization(xgb_evaluate, {'max_depth': (3,9), 
                                             'gamma': (0, 1),
                                             'colsample_bytree': (0.3,0.9)})

xgb_bo.maximize(init_points=5, n_iter=10, acq='ei')
gc.collect()

#Sorting the results so that the params with least rmse are the last element
sorted_res = sorted(xgb_bo.res,key = lambda x: x['target'])
params = sorted_res[-1]
params['params']['max_depth'] = int(params['params']['max_depth']) #max_depth should be an integer
gc.collect()

#The models list contains all the models that are trained. We will then use the mean prediction of all estimates for the final model
models = []
for i in range(n_models):
    model = xgb.XGBRegressor(colsample_bytree=params['params']['colsample_bytree'], gamma = params['params']['gamma'], max_depth=params['params']['max_depth'],n_estimators = 100)
    model.fit(Xs[i],ys[i])
    models.append(model)
    
gc.collect()

#preds will have a numpy array of predictions from each model
preds = []
for i in range(n_models):
    preds.append(models[i].predict(X_test))

preds = np.asarray(preds)
bagging_pred = np.mean(preds,axis = 0)

#create a dataframe in the submission format
holdout = pd.DataFrame({'key': test_df.key, 'fare_amount': bagging_pred})
#write the submission file to output
holdout.to_csv('submission.csv', index=False)

holdout.head()

len(holdout)