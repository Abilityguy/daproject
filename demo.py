
import lightgbm as lgb
import pandas as pd
import holidays
import numpy as np
us_holidays = holidays.US()
test_df = pd.read_feather("test_df.feather")
test_df = test_df.drop(['key','pickup_datetime'],axis = 1)
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

def getLGB():
    #kaggle kernel - https://www.kaggle.com/anushkini/taxi-lightgbm?scriptVersionId=23609067
    try:
        #try to read the model
        model = lgb.Booster(model_file = "model.txt" )
    except Exception:
        #if the trained model is not present, train it again
        lgbm_params =  {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'nthread': 4,
            'learning_rate': 0.05,
            'bagging_fraction': 1,
            'num_rounds':50000
            }
        model = lgb.train(lgbm_params, train_set = dtrain, num_boost_round=10000,early_stopping_rounds=500,verbose_eval=500, valid_sets=dval)
        del(X_train)
        del(y_train)
        del(X_val)
        del(y_val)
        gc.collect()
    
    return model

import requests
import json
from datetime import datetime

#PLEASE TURN INTERNET ON FOR THIS TO WORK... ------->

#read the details
print("Enter the source: ")
place = input()
print("Enter the destination: ")
dest = input()
print("Enter the approximate time in number of hours: ")
time = int(input())
print("Please enter the number of passengers")
psngcnt = int(input())
print("Please enter the date (dd/mm/yyyy)")
date = input().strip()

#coordinates of important places
Manhattan = (-73.9712,40.7831)[::-1]
JFK_airport = (-73.7781,40.6413)[::-1]
Laguardia_airport = (-73.8740,40.7769)[::-1]
statue_of_liberty = (-74.0445,40.6892)[::-1]
central_park = (-73.9654,40.7829)[::-1]
time_square = (-73.9855,40.7580)[::-1]
brooklyn_bridge = (-73.9969,40.7061)[::-1]
rockerfeller = (-73.9787,40.7587)[::-1]

#create a datetime object for the given day
datetime_object = datetime.strptime(date, '%d/%m/%Y')

#perform an api request in order to get the coordinates of the source and destination
response = requests.get("https://api.opencagedata.com/geocode/v1/geojson?q="+place.replace(' ', '+') +"&key=c2f9d990b75444389382e38f107441b0&pretty=1")
srccoords = json.loads(response.text)["features"][0]["geometry"]["coordinates"]
response = requests.get("https://api.opencagedata.com/geocode/v1/geojson?q="+dest.replace(' ', '+') +"&key=c2f9d990b75444389382e38f107441b0&pretty=1")
dstcoords = json.loads(response.text)["features"][0]["geometry"]["coordinates"]

#create a new dataframe for the data point
newdf = pd.DataFrame(columns = test_df.columns)
#create a new row with the extra feature
row = [srccoords[0], 
       srccoords[1],
       dstcoords[0],
       dstcoords[1],
       psngcnt,
       time*60,
       1 if datetime_object.strftime('%d-%m-%y')in us_holidays else 0,
       haversine_distance(srccoords[1], srccoords[0], dstcoords[1], dstcoords[0]),
       datetime_object.year,
       datetime_object.weekday(),
       haversine_distance(Manhattan[0],Manhattan[1],srccoords[1],srccoords[0]),
       haversine_distance(Manhattan[0],Manhattan[1],dstcoords[1],dstcoords[0]),
       haversine_distance(JFK_airport[0],JFK_airport[1],dstcoords[1],dstcoords[0]),
       haversine_distance(JFK_airport[0],JFK_airport[1],srccoords[1],srccoords[0]),
       haversine_distance(Laguardia_airport[0],Laguardia_airport[1],srccoords[1],srccoords[0]),
       haversine_distance(Laguardia_airport[0],Laguardia_airport[1],dstcoords[1],dstcoords[0]),
       datetime_object.day,
       datetime_object.month,
       haversine_distance(statue_of_liberty[0],statue_of_liberty[1],srccoords[1],srccoords[0]),
       haversine_distance(statue_of_liberty[0],statue_of_liberty[1],dstcoords[1],dstcoords[0]),
       haversine_distance(central_park[0],central_park[1],srccoords[1],srccoords[0]),
       haversine_distance(central_park[0],central_park[1],dstcoords[1],dstcoords[0]),
       haversine_distance(time_square[0],time_square[1],srccoords[1],srccoords[0]),
       haversine_distance(time_square[0],time_square[1],dstcoords[1],dstcoords[0]),
       haversine_distance(brooklyn_bridge[0],brooklyn_bridge[1],srccoords[1],srccoords[0]),
       haversine_distance(brooklyn_bridge[0],brooklyn_bridge[1],dstcoords[1],dstcoords[0]),
       haversine_distance(rockerfeller[0],rockerfeller[1],srccoords[1],srccoords[0]),
       haversine_distance(rockerfeller[0],rockerfeller[1],dstcoords[1],dstcoords[0])
      ]

print(row)
#add the row to the dataframe
newdf.loc[len(newdf)] = row
print(newdf)
time*=60

mincost = float('inf')
maxcost = 0
mintime = 0

model = getLGB()

def hours_and_minutes(time):
    hours = (time//60)
    minutes = time - hours*60
    return str(hours)+":"+str(minutes)

#find the best time
for i in range(120):
    newtime = min((max((time-60+i, 0)), 1339)) #taking care of exceptions
    newdf.loc[0, "time"] = newtime
    cost = model.predict(newdf)[0]
    if cost >maxcost:
        maxcost = cost
    if cost < mincost:
        mincost = cost
        mintime = newtime
print("The best time to leave is ", hours_and_minutes(mintime))
print("It will cost you: ", mincost, "USD")
print("savings(best case) in USD:", maxcost-mincost)
