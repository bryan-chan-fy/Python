# -*- coding: utf-8 -*-
"""
Created on Sun May 31 00:59:23 2020

@author: bryan79
"""

import pandas as pd
import os
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

#function calculate sum of error square between Calc.Bond & dirty price
def func(B,nextPayT,MatureT,c,Bprice,tau):
    #nextPayT=thistuple[0]
    #MatureT=thistuple[1]
    #c=thistuple[2]
    #tau=thistuple[3]
    #Bprice=thistuple[4]
    m=(MatureT-nextPayT)/0.5+1
    Ni=len(Bi)  #number of Bi
    Price=np.zeros(Ni)
    Price.shape=(Ni,1)
    sum_price_diffsq=0
    for i in range(0,Ni):
        Price[i]=0
        for j in range(0,int(m[i])):
            t=nextPayT[i]+j*0.5
            ttau=t/tau    #/tau
            etau=np.exp(-ttau)
            Price[i]=Price[i]+ 0.5*c[i]*np.exp((-B[0]-(B[1]-B[0])*(1-etau)/ttau-B[2]*((1-etau)/ttau-etau))*t)
        Price[i]=Price[i]+100*np.exp((-B[0]-(B[1]-B[0])*(1-etau)/ttau-B[2]*((1-etau)/ttau-etau))*t)
        sum_price_diffsq=sum_price_diffsq+(Price[i]-Bprice[i])**2
    return sum_price_diffsq

Data=pd.read_csv('EuroTreasYield_Nelson.csv')

NextPayTime=Data['NextPayTime']
TimeToT=Data['TimeToT']
c_rate=Data['c_rate']
Bi=Data['Bi']

    
nextPayT=NextPayTime.values
MatureT=TimeToT.values
c=c_rate.values*100
Bprice=Bi.values
    

#initialize B[] parameters
b0=[0.1,0.2,0.1]
B=np.zeros(3)

# boundary constraints B[0]>0, B[1]>0
bnds=((0,None),(0,None),(None,None))


result=np.zeros(50*5)
result.shape=(50,5)

# Loop through 50 tau values & optimize for each specific tau
for l in range(0,50,1):
    tau=l/10
    thistuple=(nextPayT,MatureT,c,Bprice,tau)  
    print(tau)
    res=minimize(func,b0,args=thistuple,method='L-BFGS-B',bounds=bnds,tol=1e-6)
    print(res.fun)
    print(res.x)
    # store optimized results
    result[l,0]=tau
    result[l,1]=res.fun
    result[l,2]=res.x[0]
    result[l,3]=res.x[1]
    result[l,4]=res.x[2]
    
plt.plot(result[:,0],result[:,1])  # plot sums of error sq.# plot sums of error s vs tau vs tau
       
# Pick the a particular tau and calc Bond Price & plot for comparison

Ni=len(Bi)
m=(MatureT-nextPayT)/0.5+1
bindex=20
Bb=[result[bindex,2],result[bindex,3],result[bindex,4]]
csum_price_diffsq=0
cPrice=np.zeros(Ni)
cPrice.shape=(Ni,1)
tau=bindex/10
for i in range(0,Ni):
    cPrice[i]=0
    for j in range(0,int(m[i])):
        t=nextPayT[i]+j*0.5
        ttau=t/tau    #/tau
        etau=np.exp(-ttau)
        cPrice[i]=cPrice[i]+ 0.5*c[i]*np.exp((-Bb[0]-(Bb[1]-Bb[0])*(1-etau)/ttau-Bb[2]*((1-etau)/ttau-etau))*t)
    cPrice[i]=cPrice[i]+100*np.exp((-Bb[0]-(Bb[1]-Bb[0])*(1-etau)/ttau-Bb[2]*((1-etau)/ttau-etau))*t)
    csum_price_diffsq=csum_price_diffsq+(cPrice[i]-Bprice[i])**2
print(csum_price_diffsq)
    
plt.plot(MatureT,Bprice,label='dirty price')
plt.plot(MatureT,cPrice,label='Nelson Siegel model fit')
plt.xlabel('Time to Maturity')
plt.ylabel('Bond Price')
plt.legend(loc='lower right')

    
    