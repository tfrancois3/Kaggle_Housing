# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 20:53:30 2018

@author: FRANCOIS Thomas
"""

## Importation of modules

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

## EDA

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

#Stripplot of #rooms and price
sns.stripplot(x='TotRmsAbvGrd',y='SalePrice',data=training,jitter=True,split=True)
#Barplot of #rooms and price 
ax = plt.subplot()
ax.set_ylabel('Average price')
training.groupby('TotRmsAbvGrd').mean()['SalePrice'].plot(kind='bar',figsize=(15,8), ax = ax)

#Stripplot of #Neighborhood and price
sns.stripplot(x='Neighborhood',y='SalePrice',data=training,jitter=True,split=True)

#Stripplot of #Functional and price
sns.stripplot(x='PoolArea',y='SalePrice',data=training,jitter=True,split=True)

#Comments : no visual correlation with Month sold, Year sold, Functionnality and Conditions1 or 2
#Comments : no visual correalation with PoolArea
#Problem of the 2 or 3D visualization is that does not show the relation features-saleprice

## Feature Engineering
 
#Let's select the following data that for makes sense from a graphical or logical perspective :
#LotArea
#MSSubClas
#MSZoning
#LotShape
#YearBuilt 
#Neighborhood
#TotRmsAbvGrd (+ Bathroom)
#Yearsold
#OverallCond

#Loading the data
def get_combined_data():

    # reading train data
    training = pd.read_csv('C:/Users\FRANCOIS Thomas/Documents/Code/Kaggle/train_house_price.csv')
    
    # reading test data
    testing = pd.read_csv('C:/Users\FRANCOIS Thomas/Documents/Code/Kaggle/test_house_price.csv')


    # merging train data and test data for future feature engineering
    for features in ['LotArea','MSSubClass','MSZoning','LotShape','YearBuilt','Neighborhood','TotRmsAbvGrd','FullBath','YrSold','OverallCond']:
        data[features]=training[features]+testing[features]
        
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    
    return data

data=get_combined_data()
targets = training['SalePrice']




#Creation of the variable "Number_rooms" including bathrooms


def process_MSZoning():
    
    global combined
    # we clean the Name variable
    combined.drop('MSZoning',axis=1,inplace=True)
    
    # encoding in dummy variable
    titles_dummies = pd.get_dummies(combined['MSZoning'],prefix='Title')
    combined = pd.concat([combined,titles_dummies],axis=1)
    
    # removing the title variable
    combined.drop('Title',axis=1,inplace=True)
    
    status('names')

def process_family():
    
    global combined
    # introducing a new feature : the number total of rooms including bathroom
    combined['#Rooms'] = combined['TotRmsAbvGrd'] + combined['FullBath']+ combined['HalfBath']
    

process_family()

#Dummy process



## Modeling


#Feature selection 
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(training, targets)







