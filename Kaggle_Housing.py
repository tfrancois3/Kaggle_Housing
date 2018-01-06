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
    training_int= pd.DataFrame()
    testing_int=pd.DataFrame()
    for features in ['LotArea','MSSubClass','MSZoning','LotShape','YearBuilt','Neighborhood','TotRmsAbvGrd','FullBath','HalfBath','YrSold','OverallCond']:
        training_int[features]=training[features]
        testing_int[features]=testing[features]
        
    combined=training_int.append(testing_int)  
    combined.reset_index(inplace=True)
    combined.drop('index', inplace=True, axis=1)
    
    return combined

combined=get_combined_data()
targets = training['SalePrice']

#Clean the data 
#combined.drop(combined[])

#Dummy process for variable "MSZoning"
def process_MSZoning():
    
    global combined
    
    # encoding in dummy variable
    titles_dummies = pd.get_dummies(combined['MSZoning'],prefix='MSZoning')
    combined = pd.concat([combined,titles_dummies],axis=1)
    
    # we clean the Name variable
    combined.drop('MSZoning',axis=1,inplace=True)
    
process_MSZoning()



#Dummy process for variable "LotShape"

def process_LotShape():
    
    global combined
    
    # encoding in dummy variable
    titles_dummies = pd.get_dummies(combined['LotShape'],prefix='LotShape')
    combined = pd.concat([combined,titles_dummies],axis=1)
    
    # we clean the Name variable
    combined.drop('LotShape',axis=1,inplace=True)
    
process_LotShape()


#Dummy process for variable "Neighborhood"
def process_Neighborhood():
    
    global combined
    
    # encoding in dummy variable
    titles_dummies = pd.get_dummies(combined['Neighborhood'],prefix='Neighborhood')
    combined = pd.concat([combined,titles_dummies],axis=1)
    
    # we clean the Name variable
    combined.drop('Neighborhood',axis=1,inplace=True)
    
process_Neighborhood()


#Creation of the variable "Number_rooms" including bathrooms and garage
def process_family():
    
    global combined
    # introducing a new feature : the number total of rooms including bathroom
    combined['#Rooms'] = combined['TotRmsAbvGrd'] + combined['FullBath']
    

process_family()




## Modeling



#Recovering training, testing 
def recover_train_test_target():
    global combined
    
    
    training = combined.head(1460)
    testing = combined.iloc[1460:]
    
    return training, testing

training, testing = recover_train_test_target()

#Feature selection 

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(training, targets)

#Let's have a look at the importance of each feature.
features = pd.DataFrame()
features['feature'] = training.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)

#Let's plot the importance of features
features.plot(kind='barh', figsize=(20, 20))

#Comments : as expected the most important features are :
#lotarea, yearbuilt, yearsold, overallconditions, #Rooms, MSSubClass

#Let's reduce training data
model = SelectFromModel(clf, prefit=True)
training_reduced = model.transform(training)
training_reduced.shape

#Let's reduce testing data
model = SelectFromModel(clf, prefit=True)
testing_reduced = model.transform(training)
testing_reduced.shape

#Hyperparameters tuning
run_gs=False

if run_gs:
    parameter_grid = {
                 'max_depth' : [4, 6, 8],
                 'n_estimators': [50, 10],
                 'max_features': ['sqrt', 'auto', 'log2'],
                 'min_samples_split': [1, 3, 10],
                 'min_samples_leaf': [1, 3, 10],
                 'bootstrap': [True, False],
                 }
    forest = RandomForestClassifier()
    cross_validation = StratifiedKFold(targets, n_folds=5)

    grid_search = GridSearchCV(forest,
                               scoring='accuracy',
                               param_grid=parameter_grid,
                               cv=cross_validation)

    grid_search.fit(training, targets)
    model = grid_search
    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
else: 
    parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50, 
                  'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}
    
    model = RandomForestClassifier(**parameters)
    model.fit(training, targets)

#Compute score
def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)

compute_score(model, training, targets, scoring='accuracy')
