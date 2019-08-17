# -*- coding: utf-8 -*-
# import  Libraries

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
#import dataset
dataset=pd.read_csv('Data.csv')
#Matrix for Independent Infromation
X=dataset.iloc[:,:-1].values
#Matrix for Dependent Infromation
Y=dataset.iloc[:,3].values
#Finding Missing Data Mean Strategy
from sklearn.preprocessing import Imputer # importing imputer
imputer = Imputer(missing_values="NaN",strategy="mean",axis=0) # intizaling Imputer with Strategy  and axis
imputer = imputer.fit(X[:,1:3])#fit imputer to X 
X[:,1:3]=imputer.transform(X[:,1:3])# applying imputer to X
from sklearn.preprocessing  import LabelEncoder, OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
onehotencoder =OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()
labelencoder_y=LabelEncoder()
Y=labelencoder_y.fit_transform(Y)
#Splliting the data set  into Traning  Set and Test Set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y ,test_size=0.2,random_state=0)
#putting values on feature Scale 
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)




