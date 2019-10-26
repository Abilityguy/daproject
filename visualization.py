#Visualization link - https://www.kaggle.com/anushkini/nyc-taxi-fare-graphs

#Importing modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc

%matplotlib inline


train_df= pd.read_feather('../input/kernel318ff03a29/nyc_taxi_data_raw.feather')
gc.collect() #used to flush garbage to clear ram

#We will now take a quick look at the data.
train_df.head()

#The describe() function of pandas gives us a quick summary of our data including basic statistics like mean, max, min for each column.
train_df.describe()

#We will now take a look at the fare_amount column of the dataset. This is the column to be predicted in the test set.
#Let us plot a histogram of the fare_amount frequency for each fare. We notice that the maximum fare is 250 dollars. Hence, a 250 bin histogram for each fare_amount from 0 to 250 dollars.
train_df.hist(column='fare_amount',bins = 250,figsize = (25,10))


#Seems like the frequency of fares above $100 is very low
print("Number of fares greater than 100$: ",len(train_df[train_df['fare_amount'] > 100]))
print("Total rows: ",len(train_df))

train_df[train_df['fare_amount'] <= 100].hist(column='fare_amount',bins = 100,figsize = (25,10))

#Since the data size is too long we will now take a sample for the rest of the plots
chunksize = 5_000_000
sample_df = train_df[0:chunksize]

#The NYC longitude runs from -74.03 to -73.75 while the latitude runs from 40.63 to 40.85

city_long_border = (-74.03, -73.75)
city_lat_border = (40.63, 40.85)

sample_df.plot(kind='scatter', x='dropoff_longitude', y='dropoff_latitude',
                color='red', 
                s=.02, alpha=.6,figsize=(10,10))

plt.ylim(city_lat_border)
plt.xlim(city_long_border)
plt.imshow(plt.imread('https://github.com/WillKoehrsen/Machine-Learning-Projects/blob/master/images/nyc_-74.1_-73.7_40.6_40.85.PNG?raw=true%27'), zorder = 0, extent = (-74.1, -73.7, 40.6, 40.85))

city_long_border = (-74.03, -73.75)
city_lat_border = (40.63, 40.85)

sample_df.plot(kind='scatter', x='pickup_longitude', y='pickup_latitude',
                color='blue', 
                s=.02, alpha=.6,figsize=(10,10))

plt.ylim(city_lat_border)
plt.xlim(city_long_border)
plt.imshow(plt.imread('https://github.com/WillKoehrsen/Machine-Learning-Projects/blob/master/images/nyc_-74.1_-73.7_40.6_40.85.PNG?raw=true%27'), zorder = 0, extent = (-74.1, -73.7, 40.6, 40.85))

sample_df.describe()

#Let us now check how the passenger_count for each trip varies
#We know that passenger_count varies from 0 to 7. Hence, a historgam of 7 bins will cover this.
sample_df['passenger_count'].value_counts().plot.bar(figsize = (20,10))

print("Number of 0 passenger trips: ",len(train_df[train_df['passenger_count'] == 0]))

#Now let us check the correlation beteen passenger_count and fare_amount
sample_df.plot(kind = 'scatter',color = 'green', x = 'passenger_count', y='fare_amount',figsize = (10,10))

sample_df.groupby("year")['fare_amount'].mean()

sample_df.groupby("year")['fare_amount'].mean().plot.bar(figsize = (20,10))

sample_df.groupby("weekday")['fare_amount'].mean()

sample_df.groupby("weekday")['fare_amount'].mean().plot.bar(figsize = (25,10))

sample_df.groupby("time")['fare_amount'].mean().plot(figsize = (25,10))