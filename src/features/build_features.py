import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
import os
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import pickle
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import AdaBoostRegressor
import os




#Load
base_path = r"C:\Users\imate\Documents\24.9.Notebooks_training\Energy-predictor\data\1_interim"
train_file = os.path.join(base_path, "1_cleaned_train.csv")
train_df=pd.read_csv(train_file)


location_df= pd.DataFrame()
location_df['site_id']= np.arange(0,16)
location_df['city']= ['Orlando', 'Heathrow', 'Tempe', 'Washington', 'Berkeley', 'Southampton', 'Washington', 'Ottowa', 'Orlando', 'Austin', 'Saltlake',\
                  'Ottowa', 'Dublin', 'Minneapolis', 'Philadelphia', 'Rochestor']

location_df['country']= ['US', 'UK', 'US', 'US', 'US', 'UK', 'US', 'Canada', 'US', 'US', 'US', 'Canada', 'Ireland', 'US', 'US', 'US']
train_df=train_df.merge(location_df, on='site_id', how='left')

UK=[]
US=[]
CA=[]
IRE=[]

for ptr in holidays.UnitedKingdom(years = 2016).keys():
    UK.append(str(ptr))
for ptr in holidays.UnitedKingdom(years = 2017).keys():
    UK.append(str(ptr))
for ptr in holidays.UnitedKingdom(years = 2018).keys():
    UK.append(str(ptr))
UK.append('2019-01-01')


for ptr in holidays.UnitedStates(years = 2016).keys():
    US.append(str(ptr))
for ptr in holidays.UnitedStates(years = 2017).keys():
    US.append(str(ptr))
for ptr in holidays.UnitedStates(years = 2018).keys():
    US.append(str(ptr))
US.append('2019-01-01')


for ptr in holidays.Canada(years = 2016).keys():
    CA.append(str(ptr))
for ptr in holidays.Canada(years = 2017).keys():
    CA.append(str(ptr))
for ptr in holidays.Canada(years = 2018).keys():
    CA.append(str(ptr))
CA.append('2019-01-01')


for ptr in holidays.Ireland(years = 2016).keys():
    IRE.append(str(ptr))
for ptr in holidays.Ireland(years = 2017).keys():
    IRE.append(str(ptr))
for ptr in holidays.Ireland(years = 2018).keys():
    IRE.append(str(ptr))
IRE.append('2019-01-01')

def holiday_filler(df):
  df['isHoliday']=[0]*(df.shape[0])
  df.loc[df['country']=='US', 'isHoliday']= (df['timestamp'].dt.date.astype('str').isin(US)).astype('int')
  df.loc[df['country']=='UK', 'isHoliday']= (df['timestamp'].dt.date.astype('str').isin(UK)).astype('int')
  df.loc[df['country']=='Canada', 'isHoliday']= (df['timestamp'].dt.date.astype('str').isin(CA)).astype('int')
  df.loc[df['country']=='Ireland', 'isHoliday']= (df['timestamp'].dt.date.astype('str').isin(IRE)).astype('int')

  return df

# Convert 'timestamp' column to datetime format
train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])

# Create the 'season' column based on the 'timestamp' column
train_df['season'] = train_df['timestamp'].apply(lambda x: 'Spring' if x.month in [3, 4, 5]
                                                  else 'Summer' if x.month in [6, 7, 8]
                                                  else 'Autumn' if x.month in [9, 10, 11]
                                                  else 'Winter')


train_df['IsDayTime']= train_df['timestamp'].apply(lambda x: 1 if x.hour >=6 and x.hour <=18 else 0)

train_df['relative_humidity']= 100*((np.exp((17.67*train_df['dew_temperature'])/
                                            (243.5+train_df['dew_temperature'])))/(np.exp((17.67*train_df['air_temperature'])/
                                                                                          (243.5+train_df['air_temperature']))))
train_df['square_feet']=np.log1p(train_df['square_feet'])
#Output
#train_df.to_csv(r'C:\Users\imate\Documents\24.9.Notebooks_training\Energy-predictor\data\2_processed\2_fe_train.csv', index=False)

##########################################################################################
test_file = os.path.join(base_path, "1_cleaned_test.csv")
test_df=pd.read_csv(test_file)
test_df['timestamp'] = pd.to_datetime(test_df['timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

#adding features in test data
test_df['season']= test_df['timestamp'].apply(lambda x: 'Spring' if x.month==3 or x.month==4 or x.month==5 else 'Summer' if x.month==6 or x.month==7 or x.month==8 
                                                else 'Autumn' if x.month==9 or x.month==10 or 
                                                x.month==11 else 'Winter')
test_df['IsDayTime']= test_df['timestamp'].apply(lambda x: 1 if x.hour >=6 and x.hour <=18 else 0)
test_df['relative_humidity']= 100*((np.exp((17.67*test_df['dew_temperature'])/(243.5+test_df['dew_temperature'])))/(np.exp((17.67*test_df['air_temperature'])/(243.5+test_df['air_temperature']))))
test_df=test_df.merge(location_df, on='site_id', how='left')
test_df= holiday_filler(test_df)
test_df['square_feet']=np.log1p(test_df['square_feet'])
test_df.drop(['city','country'], axis=1, inplace=True)
#test_df.to_csv(r'C:\Users\imate\Documents\24.9.Notebooks_training\Energy-predictor\data\2_processed\2_fe_test.csv', index=False)

###################Preprocessing
y_pred_df= train_df.groupby(['site_id', 'primary_use'])['meter_reading'].mean().reset_index()
train_df=train_df.sort_values(by='timestamp')
X_train, X_cv= train_test_split(train_df, test_size=0.20, shuffle=False)

#y_pred for baseline
y_pred_df.rename(columns={"meter_reading": "y_pred_base"}, inplace=True)
X_train= X_train.merge(y_pred_df, on=['site_id', 'primary_use'], how='left')
X_cv= X_cv.merge(y_pred_df,on=['site_id', 'primary_use'], how='left')
#Label encoding primary use variables
from sklearn.preprocessing import LabelEncoder

label_enc= LabelEncoder()
label_enc.fit(train_df['primary_use'])
X_train['primary_use']= label_enc.transform(X_train['primary_use'])
X_cv['primary_use']= label_enc.transform(X_cv['primary_use'])
label_enc.fit(test_df['primary_use'])
test_df['primary_use']= label_enc.transform(test_df['primary_use'])
label_enc.fit(train_df['season'])

X_train['season']= label_enc.transform(X_train['season'])
X_cv['season']= label_enc.transform(X_cv['season'])
test_df['season']= label_enc.transform(test_df['season'])
y_readings_tr=X_train['meter_reading']
y_readings_cv=X_cv['meter_reading']

X_train.drop(['meter_reading', 'y_pred_base'], axis=1, inplace=True)
X_cv.drop(['meter_reading', 'y_pred_base'], axis=1, inplace=True)

from statsmodels.stats.outliers_influence import variance_inflation_factor
s=X_train._get_numeric_data()
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(s.values, i) for i in range(s.shape[1])]
vif["features"] = s.columns

import pyarrow 
X_train.drop(['site_id', 'dew_temperature','timestamp','year','dayofyear','city','country'],axis=1,inplace=True)
X_cv.drop(['site_id', 'dew_temperature','timestamp','year','dayofyear','city','country'],axis=1,inplace=True)
test_df.drop(['site_id', 'dew_temperature','timestamp','year','dayofyear'],axis=1,inplace=True)
#Saving final data in files
test_df.to_feather(r'C:\Users\imate\Documents\24.9.Notebooks_training\Energy-predictor\data\2_processed\2_test.ftr')
#Converting pandas series to numpy
y_readings_tr=y_readings_tr.to_numpy()
y_readings_cv=y_readings_cv.to_numpy()
X_train.to_feather(r'C:\Users\imate\Documents\24.9.Notebooks_training\Energy-predictor\data\2_processed\2_X_train_new.ftr')
X_cv.to_feather(r'C:\Users\imate\Documents\24.9.Notebooks_training\Energy-predictor\data\2_processed\2_X_cv_new.ftr')
np.save(r'C:\Users\imate\Documents\24.9.Notebooks_training\Energy-predictor\reports\y_readings_tr_new.npy', y_readings_tr)
np.save(r'C:\Users\imate\Documents\24.9.Notebooks_training\Energy-predictor\reports\y_readings_cv_new.npy',y_readings_cv)