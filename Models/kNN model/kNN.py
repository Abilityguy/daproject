import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O 


#kaggle kernel link:: https://www.kaggle.com/pradyu99914/nyc-taxi-fare-models
#please refer to the commit #14.
#result: this 1NN model gives an RMSE rate of about 3.5!

#kaggle code
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


#all imports that are needed
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn import linear_model
import numpy as np
import gc 
from tqdm import tqdm
from sklearn.neighbors import KNeighborsRegressor

#read the test and train sets
train_df= pd.read_feather('../input/kernel318ff03a29/nyc_taxi_data_raw.feather')
test_df = pd.read_feather('../input/kernel318ff03a29/test_feature.feather')

#examine the dataset's fist 5 rows
print(len(train_df))
train_df.head()

#select all rows, and all columns after the second column
X_test = test_df.iloc[:,2:]
#reorder the columns of the tst set
Xt = train_df.iloc[:5,3:]
X_test = X_test[Xt.columns]
Xt.head()
X_test.head()
import gc
gc.collect()
X_test.head()


#Code to check the best value of k for this problem

#array to keep the k values
ks = []
#array to keep the error rates
errs = []
#just to prevent memerr
gc.collect()

#take 10,000 elements from the train set to test the model, since we do not have the actual values of the acual test set(kaggle competition)
X_test = train_df.iloc[len(train_df)-10000:,3:]
y_test_real = train_df.iloc[len(train_df)-10000:]["fare_amount"]


print(len(X_test))
#reorder the columns
Xt = train_df.iloc[:5,3:]
X_test = X_test[Xt.columns]
Xt.head()
X_test.head()
#reduce garbage!
gc.collect()

#loop through all the required values of k
#here, since the dataset is too big, we can not train a knn model on the whole dataset
#we have used bagging to train an ensemble of kNN models on different parts of the dataset. 
for k in tqdm(range(1, 20)):
    knnregressoroutputs = []
    for i in tqdm(range(len(train_df)//1000000)):
        #define a new model
        neigh = KNeighborsRegressor(n_neighbors=k)
        #select a chunk to train on
        df_chunk = train_df.iloc[i*10**6:(i+1)*10**6, :]
        X = df_chunk.iloc[:,3:]
        #target variable
        y = df_chunk['fare_amount']
        #fir it to this chunk
        neigh.fit(X,y)
        #get the predictions
        y_test = neigh.predict(X_test)
        #append this to the regressor outputs(this is the bagging step)
        knnregressoroutputs.append(y_test)
        gc.collect()
    
    #take a vote(average) of all the models
    res = knnregressoroutputs[0]
    for i in knnregressoroutputs[1:]:
        res+=i
    res/=len(knnregressoroutputs)
    #find the error rate for this value of k and append it to a list
    ks.append(k)
    errs.append(np.sqrt(((res-y_test_real) ** 2).mean()))

#plot the error rate and find the best k
import matplotlib.pyplot as plt
plt.plot(ks, errs)
plt.xlabel("K")
plt.ylabel("RMSE error on the test set")
plt.title("k vs error rate on the test set for kNN using bagging")
print(errs)
print(ks)

#from the graph, it is clear that k=1 is the best model.
#this is probably because of the sheer amount of data we have.

#use bagging again, but with just j=1, since that is the best model.
#for explanation of each line, please refer to the code above(its identical)
knnregressoroutputs = []
for i in tqdm(range(len(train_df)//1000000)):
    neigh = KNeighborsRegressor(n_neighbors=2)
    df_chunk = train_df.iloc[i*10**6:(i+1)*10**6, :]
    X = df_chunk.iloc[:,3:]
    #target variable
    y = df_chunk['fare_amount']
    neigh.fit(X,y)
    y_test = neigh.predict(X_test)
    knnregressoroutputs.append(y_test)
    gc.collect()

res = knnregressoroutputs[0]
for i in knnregressoroutputs[1:]:
    res+=i
res/=len(knnregressoroutputs)
#ready the submission file in a dataframe
holdout = pd.DataFrame({'key': test_df.key, 'fare_amount': res})
#write the submission file to output
holdout.to_csv('submission.csv', index=False)