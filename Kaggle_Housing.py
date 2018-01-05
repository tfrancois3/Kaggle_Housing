# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 20:53:30 2018

@author: FRANCOIS Thomas
"""

#Importation of modules

import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns

pd.options.display.max_columns = 100
pd.options.display.max_rows = 100

training = pd.read_csv('C:/Users/FRANCOIS Thomas/Documents/Code\Kaggle/train_house_price.csv')
testing = pd.read_csv('C:/Users\FRANCOIS Thomas/Documents/Code/Kaggle/test_house_price.csv')

print(training.shape)
print(testing.shape)

print(training.head())
print(training.describe())

#EDA

#Geom plof of price and surface
plt.figure(figsize=(15,8))
ax = plt.subplot()
ax.scatter(training['LotArea'],training['SalePrice'],c='green',s=10)
ax.set_xlabel('LotArea')
ax.set_ylabel('SalePrice ')

#Stripplot of MSSubClass and price
sns.stripplot(x='MSSubClass',y='SalePrice',data=training,jitter=True,split=True)
#Stripplot of MSZoning and price
sns.stripplot(x='MSZoning',y='SalePrice',data=training,jitter=True,split=True)

#Geom plof of price and yearbuilt
plt.figure(figsize=(15,8))
ax = plt.subplot()
ax.scatter(training['YearBuilt'],training['SalePrice'],c='green',s=10)
ax.set_xlabel('YearBuilt')
ax.set_ylabel('SalePrice ')


#Stripplot of Overall shape and price
sns.stripplot(x='LotShape',y='SalePrice',data=training,jitter=True,split=True)
#Barplot of overall shape and price 
ax = plt.subplot()
ax.set_ylabel('Average price')
training.groupby('LotShape').mean()['SalePrice'].plot(kind='bar',figsize=(15,8), ax = ax)

#Stripplot of Overall rating and price
sns.stripplot(x='OverallCond',y='SalePrice',data=training,jitter=True,split=True)
#Barplot of overall rating and price 
ax = plt.subplot()
ax.set_ylabel('Average price')
training.groupby('OverallCond').mean()['SalePrice'].plot(kind='bar',figsize=(15,8), ax = ax)

#Stripplot of #kitchen and price
sns.stripplot(x='KitchenAbvGr',y='SalePrice',data=training,jitter=True,split=True)
#Barplot of #kitchen and price 
ax = plt.subplot()
ax.set_ylabel('Average price')
training.groupby('KitchenAbvGr').mean()['SalePrice'].plot(kind='bar',figsize=(15,8), ax = ax)

#Stripplot of #bedroom and price
sns.stripplot(x='BedroomAbvGr',y='SalePrice',data=training,jitter=True,split=True)
#Barplot of #bedrooms and price 
ax = plt.subplot()
ax.set_ylabel('Average price')
training.groupby('BedroomAbvGr').mean()['SalePrice'].plot(kind='bar',figsize=(15,8), ax = ax)


#Stripplot of #Neighborhood and price
sns.stripplot(x='Neighborhood',y='SalePrice',data=training,jitter=True,split=True)

#Feature Engineering


#Loading the data
def get_combined_data():
    # reading train data
    training = pd.read_csv('C:/Users\FRANCOIS Thomas/Documents/Code/Kaggle/train_house_price.csv')
    
    # reading test data
    testing = pd.read_csv('C:/Users\FRANCOIS Thomas/Documents/Code/Kaggle/test_house_price.csv')

    # extracting and then removing the targets from the training data 
    targets = training['SalePrice']
    training.drop('SalePrice', 1, inplace=True)
    

    # merging train data and test data for future feature engineering
    combined = training.append(testing)
    combined.reset_index(inplace=True)
    combined.drop('index', inplace=True, axis=1)
    
    return combined

combined=get_combined_data()

#Creation of the variable "Number_rooms"



