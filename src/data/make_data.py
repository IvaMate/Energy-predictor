import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import holidays

def nan_fillers(df):
  air_temp_df=df.groupby(['site_id', 'day', 'month'])['air_temperature'].transform('mean')
  df['air_temperature'].fillna(air_temp_df, inplace=True)

  dew_temp_df=df.groupby(['site_id', 'day', 'month'])['dew_temperature'].transform('mean')
  df['dew_temperature'].fillna(dew_temp_df, inplace=True)

  cloud_df=df.groupby(['site_id', 'day', 'month'])['cloud_coverage'].transform('mean')
  df['cloud_coverage'].fillna(cloud_df, inplace=True)

  sea_level_df=df.groupby(['site_id', 'day', 'month'])['sea_level_pressure'].transform('mean')
  df['sea_level_pressure'].fillna(sea_level_df, inplace=True)

  precip_df=df.groupby(['site_id', 'day', 'month'])['precip_depth_1_hr'].transform('mean')
  df['precip_depth_1_hr'].fillna(precip_df, inplace=True)

  wind_dir_df=df.groupby(['site_id', 'day', 'month'])['wind_direction'].transform('mean')
  df['wind_direction'].fillna(wind_dir_df, inplace=True)

  wind_speed_df=df.groupby(['site_id', 'day', 'month'])['wind_speed'].transform('mean')
  df['wind_speed'].fillna(wind_speed_df, inplace=True)
  return df


#Load
train_df=pd.read_csv(r'C:\Users\imate\Documents\24.9.Notebooks_training\Final-pipeline\data\0_raw\0_merged_train.csv')#Remove outlier
train_df = train_df.drop(train_df[train_df['building_id'] == 1381].index)
#drop bad quality features
train_df.drop(['year_built', 'floor_count'], axis=1,inplace=True)
#Missing values and NaN values
train_df=nan_fillers(train_df)
train_df['cloud_coverage'].fillna(train_df['cloud_coverage'].median(), inplace=True)
train_df['sea_level_pressure'].fillna(train_df['sea_level_pressure'].median(), inplace=True)
train_df['precip_depth_1_hr'].fillna(train_df['precip_depth_1_hr'].median(), inplace=True)
#Output
train_df.to_csv(r'C:\Users\imate\Documents\24.9.Notebooks_training\Final-pipeline\data\1_interim\1_cleaned_train.csv', index=False)


test_df=pd.read_csv(r'C:\Users\imate\Documents\24.9.Notebooks_training\Final-pipeline\data\0_raw\0_merged_test.csv')#Remove outlier
#drop bad quality features
test_df.drop(['year_built', 'floor_count'], axis=1,inplace=True)
#Missing values and NaN values
test_df=nan_fillers(test_df)
test_df['cloud_coverage'].fillna(test_df['cloud_coverage'].median(), inplace=True)
test_df['sea_level_pressure'].fillna(test_df['sea_level_pressure'].median(), inplace=True)
test_df['precip_depth_1_hr'].fillna(test_df['precip_depth_1_hr'].median(), inplace=True)
#Output
test_df.to_csv(r'C:\Users\imate\Documents\24.9.Notebooks_training\Final-pipeline\data\1_interim\1_cleaned_test.csv', index=False)

