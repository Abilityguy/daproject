#reference- https://www.kaggle.com/pradyu99914/fork-of-fork-of-nyc-taxi-fare-models-dl-model


#package imports
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn import linear_model
import numpy as np
import gc 
from tqdm import tqdm
from sklearn.neighbors import KNeighborsRegressor



def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


#read the test and train sets
train_df = pd.read_feather('../input/kernel318ff03a29/nyc_taxi_data_raw.feather')
del train_df["pickup_datetime"]
test_df = pd.read_feather('../input/kernel318ff03a29/test_feature.feather')
del test_df["pickup_datetime"]
train_df= reduce_mem_usage(train_df)
test_df = reduce_mem_usage(test_df) 
gc.collect()

#examine the dataset's fist 5 rows
print(len(test_df))
test_df.head()

#examine the dataset
test_df.head()
#select all rows, and all columns after the second column
X = train_df.iloc[:20000000,2:]
X_test1 = train_df.iloc[53000000:,2:]
#target variable
y = train_df['fare_amount']
y = y[:20000000]
del train_df
gc.collect()

#select all rows, and all columns after the second column
X_test = test_df.iloc[:,1:]
#reorder the columns
Xt = X.iloc[:5]
X_test = X_test[Xt.columns]
X_test1 = X_test1[Xt.columns]
Xt.head()
X_test.head()
import gc
gc.collect()


X.head()

X_test.head()

import pandas as pd
from sklearn import preprocessing
#method taken from SO https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame

min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)
X_test = min_max_scaler.transform(X_test)
gc.collect()

from keras.models import Sequential
from keras.layers import Dense,Dropout

model = Sequential()
model.add(Dense(2048, input_dim = 16, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(256,  activation = 'tanh'))
model.add(Dropout(0.2))
model.add(Dense(128,  activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(64,  activation = 'tanh'))
model.add(Dropout(0.2))
model.add(Dense(32,  activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(16,  activation = 'tanh'))
model.add(Dropout(0.2))
model.add(Dense(1, activation = "linear"))
model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
history = model.fit(X,y, batch_size=2048, epochs = 30)


import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()


y_test = model.predict(X_test)
y_test
y_test1 = model.predict(X_test1)


holdout = pd.DataFrame({'key': test_df.key, 'fare_amount': list(y_test.reshape(9914))})
#write the submission file to output
holdout.to_csv('submission_dnn.csv', index=False)
holdout = pd.DataFrame({'fare_amount': list(y_test1.reshape(len(X_test1)))})
#write the submission file to output
holdout.to_csv('submission_dnn_670k.csv', index=False)
#reference - https://stackoverflow.com/questions/42763094/how-to-save-final-model-using-keras
model.save("model.h5")





