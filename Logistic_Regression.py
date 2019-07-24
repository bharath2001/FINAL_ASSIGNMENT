
#LOGISTIC REGRESSION MODEL :  Accuracy - 99.8%


#DATA PREPROCESSING

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('dataset.csv')


features = [ 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',      #selecting the features whose data type must be changed
           'Sunshine','WindGustSpeed','WindSpeed3pm','Humidity3pm',                
            'Cloud3pm','Temp3pm','RISK_MM']

for c in dataset.columns:          #changing the datatype
    if c in features:
      dataset[c] = pd.to_numeric(dataset[c], errors='coerce')
      
dataset = dataset.reindex(np.random.permutation(dataset.index))      #shuffeling the dataset

x = dataset.iloc[:, [2,3,4,5,6,8,12,14,18,20,22]]    #splitting independent features and
y = dataset.iloc[:, 23]                                  #dependent features

x.Evaporation.fillna(x.Evaporation.mean(),inplace = True)     #handling the missing values
x.Sunshine.fillna(x.Sunshine.mean(),inplace = True)
x.WindGustSpeed.fillna(x.WindGustSpeed.mean(),inplace = True)
x.WindSpeed3pm.fillna(x.WindSpeed3pm.mean(),inplace = True)
x.Humidity3pm.fillna(x.Humidity3pm.mean(),inplace = True)
x.Cloud3pm.fillna(x.Cloud3pm.mean(),inplace = True)
x.Temp3pm.fillna(x.Temp3pm.mean(),inplace = True)
x.MinTemp.fillna(x.MinTemp.mean(),inplace = True)
x.MaxTemp.fillna(x.MaxTemp.mean(),inplace = True)
x.Rainfall.fillna(x.Rainfall.mean(),inplace = True)

#x.dropna(axis=1,inplace=True)
#x.dropna(inplace=True)

df=pd.get_dummies(dataset.Location)   
data_frame=pd.concat([df,x],axis=1)    #dummy encoding the independent features

x=data_frame.iloc[:, 1:61].values   #removing the dummy variable trap

y=y.map({'Yes':1, 'No':0})    #dummy encoding the depenedent features

#splitting the test and train test by 20% and 80% respectively
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)   

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


#here 12 independent features have been considered

#Logistic Regression


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)