#kaggle kernel link: https://www.kaggle.com/pradyu99914/data-feature-engineering?scriptVersionId=21783669
#please note that there are commits after this, where we have added more faeatures
#all of this code is spread over a number of commits as the feature engineering took ~7 hours per new feature.
#but, the later commits do not have data visualizations. 

#this is the kernel we have used to perform feature engineering.
#not all of the visualizations that are required for feature engineering are here. Please refer to the file named "visualization.py"

#also, please note that gc.collect() is not a hack to increase the number of lines of code.. it is for efficient memory management

#some imports and kaggle default code
#!pip install modin[ray] #parallelized pandas..
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd 
import dask.dataframe as dd
import os
from tqdm import tqdm
import gc
import holidays
import matplotlib.pyplot as plt
import datetime as dt

#check the available input data
import os
for dirname, _, filenames in os.walk("/kaggle"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#datatypes of the columns. using float32 to reduce memory needed
traintypes = {'fare_amount': 'float32',
              'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'}

TRAIN_PATH = '../input/new-york-city-taxi-fare-prediction/train.csv'


df_list = [] # list to hold the batch dataframe
chunksize = 1000000 #we are reading 10**6 rows at a time

#haversine distance to calculate ditance between two points on earth
def haversine_distance(lat1, long1, lat2, long2):
    R = 6371  #radius of earth in kilometers
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)

    delta_phi = np.radians(lat2-lat1)
    delta_lambda = np.radians(long2-long1)

    #a = sin²((φB - φA)/2) + cos φA . cos φB . sin²((λB - λA)/2)
    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2

    #c = 2 * atan2( √a, √(1−a) )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    #d = R*c
    d = (R * c) #in kilometers

    return d


#help taken from https://www.kaggle.com/szelee/how-to-import-a-csv-file-of-55-million-rows
#only the method to read the chunks has been taken from here

for df_chunk in tqdm(pd.read_csv(TRAIN_PATH, usecols=cols, dtype=traintypes, chunksize=chunksize)):
     
    # Neat trick from https://www.kaggle.com/btyuhas/bayesian-optimization-with-xgboost
    # Using parse_dates would be much slower!
    df_chunk['pickup_datetime'] = df_chunk['pickup_datetime'].str.slice(0, 16)
    df_chunk['pickup_datetime'] = pd.to_datetime(df_chunk['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')
    
    # Can process each chunk of dataframe here
    # clean_data(), feature_engineer(),fit()
    #print("Before: ",len(df_chunk))
    df_chunk.dropna()
    #remove the rows that have coordinates outside the bounding box of the city and its nearby areas.
    mask = df_chunk['pickup_longitude'].between(-75, -73)
    mask &= df_chunk['dropoff_longitude'].between(-75, -73)
    mask &= df_chunk['pickup_latitude'].between(40, 42)
    mask &= df_chunk['dropoff_latitude'].between(40, 42)
    #remove the rows that have wrong number of passengers(negative or more than 8 passsengers)
    mask &= df_chunk['passenger_count'].between(0, 8)
    #remove rows with wrong fares(negative fares and grater than 250 USD..) and rows with fare amount = 0
    mask &= df_chunk['fare_amount'].between(0, 250)
    mask &= df_chunk['fare_amount'].gt(0)
    
    #apply this mask, which will remove all the inconsistent rows
    df_chunk = df_chunk[mask]
    #print("After: ",len(df_chunk))
    df_chunk = df_chunk.reset_index()  #make it featherable again. masking messes with the index. reset index helps remove this problem.
    mask = 0
    #recover memory!
    gc.collect()
    #add a new attribute to the dataset, the haversine distance between pickup and dropodd coordinates
    df_chunk["distance"] = pd.to_numeric(haversine_distance(df_chunk['pickup_latitude'], df_chunk['pickup_longitude'], df_chunk['dropoff_latitude'], df_chunk['dropoff_longitude']), downcast = 'float')
    gc.collect()
    #add a new column called time, which is the number of minutes since 12:00 am that day
    df_chunk["time"] = pd.to_numeric(df_chunk.apply(lambda r: r.pickup_datetime.hour*60 + r.pickup_datetime.minute, axis = 1), downcast = "unsigned")
    gc.collect()
    #print("time")
    
    #get a list of all US holidays
    us_holidays = holidays.US()
    #add a new column to the dataset, which is 1 if the row is on a holiday, or else, 0.
    df_chunk["holiday"] = pd.to_numeric(df_chunk.apply(lambda x: 1 if x.pickup_datetime.strftime('%d-%m-%y')in us_holidays else 0, axis =1), downcast = "unsigned")
    gc.collect()
    #add a column named year to the dataset
    df_chunk["year"] = df_chunk["pickup_datetime"].dt.year
    #add the weekday attribute to the dataset.
    df_chunk["weekday"] = pd.to_numeric(df_chunk["pickup_datetime"].dt.weekday, downcast = "unsigned")

    #important places
    Manhattan = (-73.9712,40.7831)[::-1]
    JFK_airport = (-73.7781,40.6413)[::-1]
    Laguardia_airport = (-73.8740,40.7769)[::-1]
 
    #add these distances
    df_chunk['pickup_distance_Mtn'] = pd.to_numeric(haversine_distance(Manhattan[0],Manhattan[1],df_chunk['pickup_latitude'],df_chunk['pickup_longitude']), downcast = 'float')
    df_chunk['dropoff_distance_Mtn'] = pd.to_numeric(haversine_distance(Manhattan[0],Manhattan[1],df_chunk['dropoff_latitude'],df_chunk['dropoff_longitude']), downcast = 'float')
    df_chunk['dropoff_distance_jfk'] = pd.to_numeric(haversine_distance(JFK_airport[0],JFK_airport[1],df_chunk['dropoff_latitude'],df_chunk['dropoff_longitude']), downcast = 'float')
    df_chunk['pickup_distance_jfk'] = pd.to_numeric(haversine_distance(JFK_airport[0],JFK_airport[1],df_chunk['pickup_latitude'],df_chunk['pickup_longitude']), downcast = 'float')
    df_chunk['pickup_distance_lg'] = pd.to_numeric(haversine_distance(Laguardia_airport[0],Laguardia_airport[1],df_chunk['dropoff_latitude'],df_chunk['dropoff_longitude']), downcast = 'float')
    df_chunk['dropoff_distance_lg'] = pd.to_numeric(haversine_distance(Laguardia_airport[0],Laguardia_airport[1],df_chunk['pickup_latitude'],df_chunk['pickup_longitude']), downcast = 'float')

    gc.collect()
    # append the chunk to list and merge all
    df_list.append(df_chunk)


#concatenate all the cunks to make the whole dataset into a single data frame
train_df = pd.concat(df_list)
#remove rows with distance = 0
mask = train_df['distance'].gt(0)
train_df = train_df[mask]
# Delete the dataframe list to release memory
del df_list
gc.collect()
# See what we have loaded
train_df.info()
#reset index, as mask messes with the index, which prevents it from being serializable
train_df = train_df.reset_index()
train_df.index = range(len(train_df.fare_amount))
#save to output.
train_df.to_feather('nyc_taxi_data_raw.feather')

#repeat the same thing for the test set.
df_chunk = pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/test.csv')
df_chunk['pickup_datetime'] = df_chunk['pickup_datetime'].str.slice(0, 16)
df_chunk['pickup_datetime'] = pd.to_datetime(df_chunk['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')
gc.collect()
df_chunk["time"] = pd.to_numeric(df_chunk.apply(lambda r: r.pickup_datetime.hour*60 + r.pickup_datetime.minute, axis = 1), downcast = "unsigned")
gc.collect()
#print("time")
us_holidays = holidays.US()
df_chunk["holiday"] = pd.to_numeric(df_chunk.apply(lambda x: 1 if x.pickup_datetime.strftime('%d-%m-%y')in us_holidays else 0, axis =1), downcast = "unsigned")
gc.collect()
Manhattan = (-73.9712,40.7831)[::-1]
JFK_airport = (-73.7781,40.6413)[::-1]
Laguardia_airport = (-73.8740,40.7769)[::-1]
df_chunk["distance"] = pd.to_numeric(haversine_distance(df_chunk['pickup_latitude'], df_chunk['pickup_longitude'], df_chunk['dropoff_latitude'], df_chunk['dropoff_longitude']), downcast = 'float')
df_chunk["year"] = df_chunk["pickup_datetime"].dt.year
df_chunk["weekday"] = pd.to_numeric(df_chunk["pickup_datetime"].dt.weekday, downcast= "unsigned")
df_chunk['pickup_distance_Mtn'] = pd.to_numeric(haversine_distance(Manhattan[0],Manhattan[1],df_chunk['pickup_latitude'],df_chunk['pickup_longitude']), downcast = 'float')
df_chunk['dropoff_distance_Mtn'] = pd.to_numeric(haversine_distance(Manhattan[0],Manhattan[1],df_chunk['dropoff_latitude'],df_chunk['dropoff_longitude']), downcast = 'float')
df_chunk['dropoff_distance_jfk'] = pd.to_numeric(haversine_distance(JFK_airport[0],JFK_airport[1],df_chunk['dropoff_latitude'],df_chunk['dropoff_longitude']), downcast = 'float')
df_chunk['pickup_distance_jfk'] = pd.to_numeric(haversine_distance(JFK_airport[0],JFK_airport[1],df_chunk['pickup_latitude'],df_chunk['pickup_longitude']), downcast = 'float')
df_chunk['pickup_distance_lg'] = pd.to_numeric(haversine_distance(Laguardia_airport[0],Laguardia_airport[1],df_chunk['dropoff_latitude'],df_chunk['dropoff_longitude']), downcast = 'float')
df_chunk['dropoff_distance_lg'] = pd.to_numeric(haversine_distance(Laguardia_airport[0],Laguardia_airport[1],df_chunk['pickup_latitude'],df_chunk['pickup_longitude']), downcast = 'float')
#check if any column got dropped(kaggle needs original number of test datapoints for evaluation)
print("before", len(df_chunk))
print("after", len(df_chunk))
print(df_chunk.head())
df_chunk.index = range(len(df_chunk.pickup_datetime)) #fix thr index, which gets messed upbecause of the masking
print("after", len(df_chunk))
df_chunk.to_feather('test_feature.feather')
#as we see, all the NaN values have been removed successfully.
train_df.isna().sum()

#read only the 2 required columns from the dataset
train_df = pd.read_feather('../input/kernel318ff03a29/nyc_taxi_data_raw.feather', columns = ['holiday', 'fare_amount'])
#selcet all the rows on public holidays in one dataframe
holiday_df = train_df[train_df['holiday']==1]
#select all the rows from normla days in another dataframe
not_holiday_df = train_df[train_df['holiday']==0]
train_df = 0
gc.collect()

#find the mean fare amount for both of these sets
holiday_average = np.mean(holiday_df['fare_amount'])
not_holiday_average = np.mean(not_holiday_df['fare_amount'])

print(holiday_average, not_holiday_average)

#as we can see, the fare amounts are noticably higher on public holidays than regular days. So, the Holiday attribute is useful...
label  = ["Holiday", "Not Holiday"]
x = [holiday_average, not_holiday_average]
def plot_bar_x():
    # this is for plotting purpose
    index = np.arange(len(label))
    plt.figure(figsize = (10,10))
    plt.bar(index, x)
    plt.xlabel('Holiday')
    plt.ylabel('Average Fare Amount')
    plt.xticks(index, label, rotation=0)
    plt.title('Average fare amounts on holidays and normal days')
    plt.show()
plot_bar_x()

#as we can see, the distance between the coordinates is a very important feature, as it is strongly related to the target variable
train_df = pd.read_feather('../input/kernel318ff03a29/nyc_taxi_data_raw.feather', columns = ['distance', 'fare_amount'])
sample_df = train_df.sample(n = 5000)
plt.figure(figsize = (10,10))
plt.scatter(sample_df["distance"], sample_df["fare_amount"])
plt.title('Scatter plot of distance vs fare_amount')
plt.xlabel('distance')
plt.ylabel('fare_amount')
plt.show()
print(len(train_df[train_df['distance']==0]))
train_df = 0
gc.collect()



