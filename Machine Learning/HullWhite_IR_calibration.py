# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 02:13:32 2021

"""

import numpy as np
import pandas as pd
import quandl
import matplotlib.pyplot as plt

y_data = quandl.get("USTREASURY/YIELD", start_date="2021-01-01", end_date="2021-07-15")
dy_data = y_data.diff()/100
dy_std=dy_data.apply(np.std,axis=0)
dy_std=dy_std*np.sqrt(252)

#df=pd.DataFrame(columns=['tau','a','sig','calc'])
df=pd.DataFrame(columns=['a','sig','calc'])
        
for j in range(1,100,1):
    for k in range(1,100,1):
        calc=0
        for i in range(1,60,1):
            tau=i*0.5
            a=0.0001*j
            sig=0.001*k
            calc+=sig*(1-np.exp(-a*tau))/(a*tau)
        #df=df.append({'tau':tau,'a':a,'sig':sig,'calc':calc},ignore_index=True)
        df=df.append({'a':a,'sig':sig,'calc':calc/59},ignore_index=True)
            
#X1=df[['tau','calc']].values
X1=df[['calc']].values       
y=df[['a','sig']].values

Y1=df[['a']].values
Y2=df[['sig']].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X1,y,test_size=0.1,random_state=101)


#X_train=X1
#Y1_train=Y1
Y1_train=y_train[:,[0]]
#Y2_train=Y2
Y2_train=y_train[:,[1]]

Y1_test=y_test[:,[0]]
Y2_test=y_test[:,[1]]


from sklearn.preprocessing import MinMaxScaler
Scaler=MinMaxScaler()
Scaler.fit(X_train)
X_train=Scaler.transform(X_train)
X_test=Scaler.transform(X_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model1=Sequential()
model1.add(Dense(32,activation='relu'))
model1.add(Dense(32,activation='relu'))
model1.add(Dense(32,activation='relu'))
#model1.add(Dense(32,activation='relu'))
model1.add(Dense(1))

model2=Sequential()
model2.add(Dense(64,activation='relu'))
model2.add(Dense(64,activation='relu'))
model2.add(Dense(64,activation='relu'))
model2.add(Dense(1))

#model1.compile(optimizer='rmsprop',loss='mse')
#model2.compile(optimizer='rmsprop',loss='mse')
from tensorflow.keras.optimizers import SGD
optimizer=SGD(lr=0.01, momentum=0.9,clipvalue=1.0)
model1.compile(optimizer=optimizer,loss='mse')
model2.compile(optimizer='adam',loss='mse')

model1.fit(x=X_train,y=Y1_train,epochs=250)
model2.fit(x=X_train,y=Y2_train,epochs=250)
#X1_test=pd.DataFrame([[0.5,0.049689],[1.0,0.04938]])

model1.evaluate(X_test,Y1_test,verbose=0)

test_pred1=model1.predict(X_test)
test_pred2=model2.predict(X_test)

X1_test=pd.DataFrame([[0.004234],[0.001825]])
X1_test=Scaler.transform(X1_test)

test_predictions1=model1.predict(X1_test)
test_predictions2=model2.predict(X1_test)