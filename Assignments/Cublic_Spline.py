# -*- coding: utf-8 -*-
"""
Created on Sun May 31 15:03:11 2020

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

out=pd.read_csv('TreasYield.csv')

R=out['Yield']
T=out['T']

n=len(T)
n_all=(n+2)*((n-1)+3)
#RT=np.transpose(R)


RT=np.zeros(n+2)
RT.shape=(n+2,1)

for i in range(0,n):
    RT[i]=R[i]
RT[n]=0     #R''(0,t1)=0
RT[n+1]=0   #R''(0,tn)=0


A=np.zeros(n_all)
A.shape=(n+2,(n-1)+3)

np.set_printoptions(precision=3)

t1=T[0]
tn=T[n-1]
for j in range(0,n):
    j
    A[j,0] = 1
    A[j,1] = T[j]-t1
    A[j,2] = (T[j]-t1)**2
    for k in range(0,n-1):
        A[j,k+3]=max(T[j]-T[k],0)**3
A[n,2]=2
A[n+1,2]=2
for k in range(0,n-1):
        A[n+1,k+3]=6*(tn-T[k])

# RT = A. C    (C=Coefficient Matrix)

Ainv=np.linalg.inv(A) # Inverse of A
C=np.dot(Ainv,RT) 

# Check the fitting by plotting yield curve

rt=np.zeros(n)
rt.shape=(n,1)
for m in range(0,n):
    tt=T[m]
    dterms=0
    for s in range(0,n-1):
        dterms=dterms+C[s+3]*(max(tt-T[s],0)**3)
    rt[m]=C[0]+C[1]*(tt-T[0])+C[2]*(tt-T[0])**2+dterms
plt.plot(T,R,'g',label='yield rate')
plt.plot(T,rt,'r-.',label='regenerated yield')
plt.xlabel('Time to Maturity')
plt.ylabel('Yield')
plt.legend(loc='lower right')

