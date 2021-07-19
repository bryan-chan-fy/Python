# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 02:13:32 2021

"""

import numpy as np
import pandas as pd
import quandl
import matplotlib.pyplot as plt
import seaborn as sns

y_data = quandl.get("USTREASURY/YIELD", start_date="2021-01-01", end_date="2021-07-15")
dy_data = y_data.diff()/100             # dR
dy_std=dy_data.apply(np.std,axis=0)     # std(dR)
dy_std=dy_std*np.sqrt(252)              # std(dR) * sqrt(252)



t1=1    # 1 yr
t2=5    # 5 yr
t3=10   # 10 yr
t4=30   # 30 yr
#inp1=0.005
#inp2=0.005
#inp3=0.005
#inp4=0.005


#df=pd.DataFrame(columns=['tau','a','sig','calc'])
#df=pd.DataFrame(columns=['a','sig','calc'])
df=pd.DataFrame(columns=['inp1','inp2','inp3','inp4','a','sig'])

for i1 in range(1,10,1):
    for i2 in range(1,10,1):
        for i3 in range(1,10,1):
            for i4 in range(1,10,1):
                inp1=i1*0.001   # input 1 std(dR)
                inp2=i2*0.001   # input 2 std(dR)
                inp3=i3*0.001   # input 3 std(dR)
                inp4=i4*0.001   # input 4 std(dR)
                Lmat=[[0 for i in range(99)] for j in range(99)]  # Store diff input std(dR) & calc.
                for j in range(1,100,1):
                    for k in range(1,100,1):
                        a=0.0001*j
                        sig=0.001*k
                        calc1=sig*(1-np.exp(-a*t1))/(a*t1)
                        calc2=sig*(1-np.exp(-a*t2))/(a*t2)
                        calc3=sig*(1-np.exp(-a*t3))/(a*t3)
                        calc4=sig*(1-np.exp(-a*t4))/(a*t4)
                        Lmat[j-1][k-1]=abs(calc1-inp1)+abs(calc2-inp2)+abs(calc3-inp3)+abs(calc4-inp4)
                min_rowind=int(np.argmin(Lmat)/99)
                min_colind=np.argmin(Lmat)%99
                a_sel=0.0001*(min_rowind+1)
                sig_sel=0.001*(min_colind+1)
                df=df.append({'inp1':inp1,'inp2':inp2,'inp3':inp3,'inp4':inp4,'a':a_sel,'sig':sig_sel},ignore_index=True)
        
#for j in range(1,100,1):
#    for k in range(1,100,1):
#        calc=0
#        for i in range(1,60,1):
#            tau=i*0.5
#            a=0.0001*j
#            sig=0.001*k
#            calc+=sig*(1-np.exp(-a*tau))/(a*tau)
#        #df=df.append({'tau':tau,'a':a,'sig':sig,'calc':calc},ignore_index=True)
#        df=df.append({'a':a,'sig':sig,'calc':calc/59},ignore_index=True)

df.to_csv(r'C:\Users\bryan79\Documents\Python files\CalibSim.csv',index=False)
            
#X1=df[['tau','calc']].values
#X1=df[['calc']].values
X1=df[['inp1','inp2','inp3','inp4']].values 
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
model1.add(Dense(64,activation='relu'))
model1.add(Dense(64,activation='relu'))
model1.add(Dense(64,activation='relu'))
#model1.add(Dense(32,activation='relu'))
model1.add(Dense(1))

model2=Sequential()
model2.add(Dense(64,activation='relu'))
model2.add(Dense(64,activation='relu'))
model2.add(Dense(64,activation='relu'))
model2.add(Dense(1))

#model1.compile(optimizer='rmsprop',loss='mse')
#model2.compile(optimizer='rmsprop',loss='mse')
from tensorflow.keras.optimizers import SGD, Adam
#optimizer=SGD(lr=0.01, momentum=0.9,clipvalue=1.0)
optimizer=Adam(lr=0.01)
model1.compile(optimizer=optimizer,loss='mse')
#model1.compile(optimizer='adam',loss='mse')
model2.compile(optimizer='adam',loss='mse')

model1.fit(x=X_train,y=Y1_train,epochs=250)
model2.fit(x=X_train,y=Y2_train,epochs=250)
#X1_test=pd.DataFrame([[0.5,0.049689],[1.0,0.04938]])

#X1_test=pd.DataFrame([[0.004234],[0.001825]])
#X1_test=Scaler.transform(X1_test)

loss_df1 = pd.DataFrame(model1.history.history)  # Training history Loss
loss_df2 = pd.DataFrame(model2.history.history)
loss_df1.plot()
loss_df2.plot()


test_predictions1=model1.predict(X_test)
test_predictions2=model2.predict(X_test)
#test_predictions1=pd.Series(test_predictions1.reshape(657,))
#test_predictions2=pd.Series(test_predictions2.reshape(657,))
#pred_df1 = pd.DataFrame(Y1_test)
#pred_df1 = pd.concat([pred_df1,test_predictions1],axis=1)
#pred_df1.columns =  ['Test True a','Model Predictions']
#pred_df2 = pd.DataFrame(Y2_test)
#pred_df2 = pd.concat([pred_df2,test_predictions2],axis=1)
#pred_df2.columns =  ['Test True sig','Model Predictions']

#sns.scatterplot(x='Test True a',y='Model Predictions',data=pred_df1)
#sns.scatterplot(x='Test True sig',y='Model Predictions',data=pred_df2)

from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score
np.sqrt(mean_squared_error(Y1_test,test_predictions1))
Y1.mean()
np.sqrt(mean_squared_error(Y2_test,test_predictions2))
Y2.mean()

plt.plot(Y1_test)
plt.plot(test_predictions1)
plt.plot(Y2_test)
plt.plot(test_predictions2)

explained_variance_score(Y1_test,test_predictions1) # Variance regression score
explained_variance_score(Y2_test,test_predictions2)

X1_test_single=pd.DataFrame([[dy_std[4],dy_std[7],dy_std[9],dy_std[11]]])  # Using Quandl data as 4 inputs
#X1_test=pd.DataFrame([[0.001397,0.005589,0.006464,0.006853]])
X1_test_single=Scaler.transform(X1_test_single)

test_predictions1_single=model1.predict(X1_test_single)
test_predictions2_single=model2.predict(X1_test_single)

