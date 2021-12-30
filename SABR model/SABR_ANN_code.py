# Import libraries

import numpy as np
import pandas as pd
import csv
import datetime
from scipy.stats import norm
from numpy import dot
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input,Model
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Parameters definition

M=160  	# Discretization max. count
N=100
dt=0.0040   	# time step
Xo=-8		# lower bound
Yo=-10
dX=(2-Xo)/M	   # Discretization steps
dY=(0-Yo)/N
beta=1.0
xi=0.1 
rho=0.5
r=0.05
T=0.5   # Maturity
P=T/dt
K=1     # Strike
Klist=[1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5]  #Strikes 
xilist=[0.005,0.05,0.1,0.3,0.5]     #xi
rholist=[-0.7,-0.4,-0.1,0.1,0.4,0.7]  #rho


# Define matrix A elements functions for Explicit equation

def au_func(i,j,n,dt,beta,rho,xi,dY,dX,Yo,Xo,r):
    Xp=Xo+i*dX
    Yp=Yo+j*dY
    return 0.25*dt*rho*xi*np.exp(Yp)*np.exp((beta-1)*(Xp))/(dX*dY)

def am_func(i,j,n,dt,beta,rho,xi,dY,dX,Yo,Xo,r):
    Xp=Xo+i*dX
    Yp=Yo+j*dY
    return 0.5*dt*(xi**2)*(1/(dY**2)-1/(2*dY))

def ad_func(i,j,n,dt,beta,rho,xi,dY,dX,Yo,Xo,r):
    Xp=Xo+i*dX
    Yp=Yo+j*dY
    return -0.25*dt*rho*xi*np.exp(Yp)*np.exp((beta-1)*(Xp))/(dX*dY)

def bu_func(i,j,n,dt,beta,rho,xi,dY,dX,Yo,Xo,r):
    Xp=Xo+i*dX
    Yp=Yo+j*dY
    return 0.5*dt*np.exp(2*Yp)*np.exp(2*(beta-1)*(Xp))*(1/(dX**2)-1/(2*dX))                                               
    
def bm_func(i,j,n,dt,beta,rho,xi,dY,dX,Yo,Xo,r):
    Xp=Xo+i*dX
    Yp=Yo+j*dY
    return 1+dt*(-np.exp(2*Yp)*np.exp(2*(beta-1)*(Xp))/(dX**2)-(xi**2)/(dY**2))  

def bd_func(i,j,n,dt,beta,rho,xi,dY,dX,Yo,Xo,r):
    Xp=Xo+i*dX
    Yp=Yo+j*dY
    return 0.5*dt*np.exp(2*Yp)*np.exp(2*(beta-1)*(Xp))*(1/(dX**2)+1/(2*dX))
    
def cu_func(i,j,n,dt,beta,rho,xi,dY,dX,Yo,Xo,r):
    Xp=Xo+i*dX
    Yp=Yo+j*dY
    return -0.25*dt*rho*xi*np.exp(Yp)*np.exp((beta-1)*(Xp))/(dX*dY)  

def cm_func(i,j,n,dt,beta,rho,xi,dY,dX,Yo,Xo,r):
    Xp=Xo+i*dX
    Yp=Yo+j*dY
    return 0.5*dt*(xi**2)*(1/(dY**2)+1/(2*dY))

def cd_func(i,j,n,dt,beta,rho,xi,dY,dX,Yo,Xo,r):
    Xp=Xo+i*dX
    Yp=Yo+j*dY
    return 0.25*dt*rho*xi*np.exp(Yp)*np.exp((beta-1)*(Xp))/(dX*dY)

def ru_func(i,dt,xi,dY,r):
    return dt*0.5*(xi**2)*(1/(dY**2)-1/(2*dY)) 

def rm_func(i,dt,xi,dY,r):    
    #return 1 #(1-r*dt)
    return 1-dt*(xi**2)/(dY**2)

def rd_func(i,dt,xi,dY,r):
return dt*0.5*(xi**2)*(1/(dY**2)+1/(2*dY)) 

#=== Computation – Finite Difference Method (FDM) for particular T ===

for oo in range(len(rholist)):   # Loop rho
    rho=rholist[oo]
    for ooo in range(len(xilist):     # Loop xi
        xi=xilist[ooo]
        n=0
        A=np.zeros(((M+1)*(N+1),(M+1)*(N+1)))
        # Setup matrix A 
        for j in range(N+1):
            for i in range(M+1):
                row=(M+1)*(N+1)-j*(M+1)-i-1
        
                if(j==0):
                    if (i==0):
                        A[row,(M+1)*(N+1)-1]=1
                    elif (i>0) & (i<M):
                        A[row,(M+1)*(N+1)-1-i]=rm_func(i,dt,xi,dY,r)
                        A[row,(M+1)*(N+1)-1-(i+1)]=ru_func(i,dt,xi,dY,r)
                        A[row,(M+1)*(N+1)-1-(i-1)]=rd_func(i,dt,xi,dY,r)
                    elif (i==M):
                        A[row,(M+1)*(N+1)-1-i]=1
                        A[row,(M+1)*(N+1)-1-i+1]=-1
                elif(j==N) & (i>0):
                    A[row,(M+1)*(N+1)-j*(M+1)-1-i]=1
                    A[row,(M+1)*(N+1)-j*(M+1)-1-i+1]=-1
                elif(i==0):
                    A[row,(M+1)*(N+1)-j*(M+1)-1]=1
                elif(i==M):
                    A[row,(M+1)*(N+1)-(j+1)*(M+1)+1-1]=1
                    A[row,(M+1)*(N+1)-(j+1)*(M+1)+2-1]=-1
                elif(i>0) & (i<M):
                    A[row,(M+1)*(N+1)-j*(M+1)-1-i]=bm_func(i,j,n,dt,beta,rho,xi,dY,dX,Yo,Xo,r)
                    A[row,(M+1)*(N+1)-j*(M+1)-1-(i+1)]=bu_func(i,j,n,dt,beta,rho,xi,dY,dX,Yo,Xo,r)
                    A[row,(M+1)*(N+1)-j*(M+1)-1-(i-1)]=bd_func(i,j,n,dt,beta,rho,xi,dY,dX,Yo,Xo,r)
            
                    A[row,(M+1)*(N+1)-(j-1)*(M+1)-1-i]=cm_func(i,j,n,dt,beta,rho,xi,dY,dX,Yo,Xo,r)
                    A[row,(M+1)*(N+1)-(j-1)*(M+1)-1-(i+1)]=cu_func(i,j,n,dt,beta,rho,xi,dY,dX,Yo,Xo,r)
                    A[row,(M+1)*(N+1)-(j-1)*(M+1)-1-(i-1)]=cd_func(i,j,n,dt,beta,rho,xi,dY,dX,Yo,Xo,r)
            
                    A[row,(M+1)*(N+1)-(j+1)*(M+1)-1-i]=am_func(i,j,n,dt,beta,rho,xi,dY,dX,Yo,Xo,r)
                    A[row,(M+1)*(N+1)-(j+1)*(M+1)-1-(i+1)]=au_func(i,j,n,dt,beta,rho,xi,dY,dX,Yo,Xo,r)
                    A[row,(M+1)*(N+1)-(j+1)*(M+1)-1-(i-1)]=ad_func(i,j,n,dt,beta,rho,xi,dY,dX,Yo,Xo,r)

        for o in range(len(Klist)):		# Loop K
            K=Klist[o]
            B=np.zeros(((M+1)*(N+1),1))
            n=0
		# Initial values t=0
            for j in range(N+1):
                for i in range(M+1):
                    row=(M+1)*(N+1)-j*(M+1)-i-1
        
                    B[row,0]=max(np.exp(Xo+i*dX)-K,0)
                    if(j==N) & (i>0):
                        B[row,0]=B[row+1,0]+dX*np.exp(Xo+i*dX)
                    if(i==M):
                        B[row,0]=B[row+1,0]+dX*np.exp(Xo+i*dX)
            
            while (n<int(P)):			# Loop time
                # Matrix Multiplication
                C = dot(A,B)  
    
                # Reinforce boundary conditions
                for j in range(N+1):
                    row1=(M+1)*(N+1)-j*(M+1)-1      # when i=0
                    row2=(M+1)*(N+1)-j*(M+1)-M-1    # when i=M
                    C[row1,0]=0
                    C[row2,0]=C[row2+1,0]+dX*np.exp(Xo+M*dX)
                for i in range(1,M+1):
                    row3=(M+1)*(N+1)-N*(M+1)-i-1    # when j=N
                    C[row3,0]=C[row3+1,0]+dX*np.exp(Xo+i*dX)
    
                B=C
                n+=1
            
		# Append/Write to CSV file - price data 
            paraml=[rho,xi,T,K,beta,dY,dX,dt,str(datetime.datetime.now())]
            wdata=B.T.tolist()[0] 
            wdata.extend(paraml)  # Combine B matrix and parameters info
            with open(r'C:\Users\Documents\Python files\SABRdata1.csv', 'a', newline='') as file:   
                writer = csv.writer(file)
                writer.writerow(wdata)


#=== Read price data, store in 2D-matrix, convert into implied vol. ===

# Read from CSV file – price data
ix=0;
dtr=[['']*((M+1)*(N+1)+9) for s in range(300)]
with open(r'C:\Users\Documents\Python files\SABRdata1.csv', 'r',) as file:
    reader = csv.reader(file)
    for rowr in reader:
        datar=rowr
        dtr[ix]=np.array(datar)
        ix+=1

# Store price data in 2-D matrix
fix_list=[0.5,1.0,1.5,2.0,2.5,3.0,3.5]    # Fo values
vix_list=[0.02,0.10,0.30,0.50]		# v0 values
N_comb=len(xilist)*len(rholist)		# number of combinations
datamat=np.zeros((N_comb,len(Klist),len(fix_list)*len(vix_list))) # data matrix


# Function return B matrix element position for a particular F0 & V0
def Findx(M,N,F0,dX,V0,dY):
    lnV0=np.log(V0)
    lnF0=np.log(F0)
    posV=np.round((lnV0-Yo)/dY)
    posF=np.round((lnF0-Xo)/dX)
    return (M+1)*(N+1)-1-int(posV)*(M+1)-int(posF)


cur_T=0.5
for ix in range(0,300):
    cur_rho=dtr[ix][(M+1)*(N+1)-1+1]
    cur_xi=dtr[ix][(M+1)*(N+1)-1+2]
    cur_K=dtr[ix][(M+1)*(N+1)-1+4]
    rho_idx = rholist.index(float(cur_rho))  # rho index position
    xi_idx= xilist.index(float(cur_xi)) # xi index position
    comb_idx=rho_idx*(len(xilist))+xi_idx  # combination index
    K_idx=Klist.index(float(cur_K))  		# K index position
        
    fi=0
    for fix in fix_list:	# Loop Fo
        vi=0
        for vix in vix_list:		# Loop v0
            pos=Findx(M,N,fix,dX,vix,dY)  # element position of B matrix
            datamat[comb_idx][K_idx][fi*len(vix_list)+vi]=dtr[ix][pos]
            vi+=1
        fi+=1            


# Function return call option price with vol. input – Black’s equation
def f(sigma):
    d1=(np.log(F0/(K+1e-99))+0.5*(sigma**2)*T)/(sigma*np.sqrt(T))
    d2=(np.log(F0/(K+1e-99))-0.5*(sigma**2)*T)/(sigma*np.sqrt(T))
    return F0*norm.cdf(d1)-(K+1e-99)*norm.cdf(d2)
    

# Dekker’s root-finding function 
def Dekker(target,a,b,tol,f):
    fa=f(a)
    fb=f(b)
    if (abs(fa-target)<abs(fb-target)):  # Swap if function of b is bigger
        c=a
        d=b
        c0=d
    else:
        c=b
        d=a
        c0=d
    
    while (abs(c-d)>tol):  # loop while bigger than tolerance

        s= c-(f(c)-target)*(c-c0)/(f(c)-f(c0))  # Secant value
        m= 0.5*(c+d)					# Bisection value
    
        if (s>m) & (s<c):
            cn=s
        else:
            cn=m
    
        c0=c
        dn=d
        if ((f(d)-target)*(f(cn)-target)>0):
            dn=c
        
        if (abs(f(dn)-target)<abs(f(cn)-target)):
            xtemp=dn
            dn=cn
            cn=xtemp

        d=dn
        c=cn
   
    return c 

# Convert price data in 2-D matrix to volatilities. Pick only Fo=3.0 & 3.5
T_=0.5
for rho_idx in range(len(rholist)): 	#Loop rho
    rho_=rholist[rho_idx]
    for xi_idx in range(len(xilist)):    #Loop xi
        xi_=xilist[xi_idx]
        for K_idx in range(len(Klist)):		#Loop K
            K_=Klist[K_idx]
            for vi in range(len(vix_list)):
                v_=vix_list[vi]
                fi1=fix_list.index(3.0)  # Fo=3.0 index
                fi2=fix_list.index(3.5)  # Fo=3.5 index
                comb_idx=rho_idx*(len(xilist))+xi_idx  #comb. index
                price1=datamat[comb_idx][K_idx][fi1*len(vix_list)+vi]
                price2=datamat[comb_idx][K_idx][fi2*len(vix_list)+vi]
                T=T_
                K=K_
                F01=np.exp(Xo+dX*np.round((np.log(3.0)-Xo)/dX)) 
                F0=F01
                vol1=Dekker(price1,0.01,1,1e-6,f) 	# compute implied vol
                F02=np.exp(Xo+dX*np.round((np.log(3.5)-Xo)/dX)) 
                F0=F02
                vol2=Dekker(price2,0.01,1,1e-6,f)	# compute implied vol

   # Append/Write imp volatility to CSV file
                param2=[rho_,xi_,T_,K_,v_,beta]
                x=[price1,price2,vol1,vol2,F01,F02] 
                x.extend(param2)
                
                with open(r'C:\Users\Documents\Python files\SABRdata_pro1.csv', 'a', newline='') as file2: 
                    writer = csv.writer(file2)
                    writer.writerow(x)


#=== Read All Implied volatility data & prepare data as Data Frame ===

# Read imp volatility from CSV files (containing T=0.5,1,2,5,10,20)
idx=0;
inpdat=[]  # Hold Implied Vol. for Fo=3.0 & 3.5
with open(r'C:\Users\Documents\Python files\SABRdata_pro1.csv', 'r',) as filedat:# T=0.5 imp. vol. data
    reader = csv.reader(filedat)
    for rowd in reader:
        datar=rowd
        inpdat.append(np.array(datar))
with open(r'C:\Users\Documents\Python files\SABRdata_pro2.csv', 'r',) as filedat:# T=1 imp. vol. data
    reader = csv.reader(filedat)
    for rowd in reader:
        datar=rowd
        inpdat.append(np.array(datar))
with open(r'C:\Users\Documents\Python files\SABRdata_pro3.csv', 'r',) as filedat:# T=2 imp. vol. data
    reader = csv.reader(filedat)
    for rowd in reader:
        datar=rowd
        inpdat.append(np.array(datar))
with open(r'C:\Users\Documents\Python files\SABRdata_pro4.csv', 'r',) as filedat:# T=5	imp. vol. data
    reader = csv.reader(filedat)
    for rowd in reader:
        datar=rowd
        inpdat.append(np.array(datar))
with open(r'C:\Users\Documents\Python files\SABRdata_pro5.csv', 'r',) as filedat:# T=10 imp. vol. data
    reader = csv.reader(filedat)
    for rowd in reader:
        datar=rowd
        inpdat.append(np.array(datar))
with open(r'C:\Users\Documents\Python files\SABRdata_pro6.csv', 'r',) as filedat:# T=20 imp. vol. data
    reader = csv.reader(filedat)
    for rowd in reader:
        datar=rowd
        inpdat.append(np.array(datar))


df=pd.DataFrame(columns=['rho','xi','T','v0','F0','vol1','vol2','vol3','vol4','vol5','vol6','vol7','vol8','vol9'])  # setup Data Frame 

fix_list=[0.5,1.0,1.5,2.0,2.5,3.0,3.5]	# Fo values                 
vix_list=[0.02,0.10,0.30,0.50]		# v0 values
Tlist=[0.5,1.0,2.0,5.0,10.0,20.0]		# T values

for Tidx in range(len(Tlist)):  # Loop T
    off_T=Tidx*len(rholist)*len(xilist)*len(vix_list)*len(Klist) #offset
    
    for rhoidx in range(len(rholist)): # Loop rho
        off_rho=rhoidx*len(xilist)*len(vix_list)*len(Klist)  #offset
        
        for xiidx in range(len(xilist)): # Loop xi
            off_xi=xiidx*len(vix_list)*len(Klist)	 #offset
            
            for v0idx in range(len(vix_list)):
                off_base=off_T+off_rho+off_xi+v0idx
                rho_1=float(inpdat[off_base][6])  # extract the parameters
                xi_1=float(inpdat[off_base][7])
                T_1=float(inpdat[off_base][8])
                v0_1=float(inpdat[off_base][10])
                
                F01_v=float(inpdat[off_base][4])
                F02_v=float(inpdat[off_base][5])
                vol_1=np.zeros(len(Klist)-1)
                vol_2=np.zeros(len(Klist)-1)
                indx=0    
                for Kidx in range(1,len(Klist)): # conglomerate 9 imp vols
                    nix1=off_base+(Kidx-1)*len(vix_list)  #index for Fo=3.0
                    nix2=off_base+Kidx*len(vix_list)	   #index for Fo=3.5
                    vol_1[indx]=float(inpdat[nix1][2]) # Imp vol for Fo=3.0
                    vol_2[indx]=float(inpdat[nix2][3]) # Imp vol for Fo=3.5
                    indx+=1
                                
   	# Append to Data Frame         df=df.append({'rho':rho_1,'xi':xi_1,'T':T_1,'v0':v0_1,'F0':F01_v,'vol1':vol_1[0],'vol2':vol_1[1],'vol3':vol_1[2],'vol4':vol_1[3],'vol5':vol_1[4],'vol6':vol_1[5],'vol7':vol_1[6],'vol8':vol_1[7],'vol9':vol_1[8]},ignore_index=True)
                df=df.append({'rho':rho_1,'xi':xi_1,'T':T_1,'v0':v0_1,'F0':F02_v,'vol1':vol_2[0],'vol2':vol_2[1],'vol3':vol_2[2],'vol4':vol_2[3],'vol5':vol_2[4],'vol6':vol_2[5],'vol7':vol_2[6],'vol8':vol_2[7],'vol9':vol_2[8]},ignore_index=True)


#=== Plot DFM implied vol. with Hagan’s implied vol ===

# Function: Hagan’s approximation for implied volatilities
def calc_impvol(rho_,xi_,V0_,T_,F0_,K_,beta_):
    q=(F0_*K_)**((1-beta_)/2)*(1+((1-beta_)**2)*(np.log(F0_/K_)**2)/24 + (((1-beta_)**4)/1920)*(np.log(F0_/K_)**4))
    z=(xi_/V0_)*((F0_*K_)**((1-beta_)/2)*np.log(F0_/K_))
    X_z=np.log((np.sqrt(1-2*rho_*z+z**2)-rho_+z)/(1-rho_))
    vol_calc= (V0_/q)*((z+1e-99)/(X_z+1e-99))*( 1+( ((1-beta_)**2)/(24*((F0_*K_)**(1-beta_))) + (rho_*xi_*beta_*V0_)/(4*(F0_*K_)**((1-beta_)/2)) + ((2-3*rho_**2)*xi_**2)/24 )*T_)
    return vol_calc


i=476 # a particular row
rho_p=float(df[['rho']].loc[i]) # Get parameter values
xi_p=float(df[['xi']].loc[i])
T_p=float(df[['T']].loc[i])
v0_p=float(df[['v0']].loc[i])
F0_p=float(df[['F0']].loc[i])
beta_p=1
yp=[df[['vol1']].loc[i],df[['vol2']].loc[i],df[['vol3']].loc[i],df[['vol4']].loc[i],df[['vol5']].loc[i],df[['vol6']].loc[i],df[['vol7']].loc[i],df[['vol8']].loc[i],df[['vol9']].loc[i]]  # DFM plot values
yhag=[]   # Hold Hagan’s plot values
for n in range(len(Klist)-1):  # Choose the correct Strikes for Fo=3.0 or Fo=3.5
    if (i%2)==0:  # Fo=3.0
        K_p=Klist[n]
    else:
        K_p=Klist[n+1]
    xpp=calc_impvol(rho_p,xi_p,v0_p,T_p,F0_p,K_p+1e-199,beta_p) # Hagan’s imp vol.
    yhag.append(xpp)

if (i%2)==0:  # Fo=3.0
plt.plot(Klist[0:9],yp[0:9], label='FD implied vol')
plt.plot(Klist[0:9],yhag[0:9], label='Hagan implied vol')
else:
	plt.plot(Klist[1:10],yp[0:9], label='FD implied vol')
plt.plot(Klist[1:10],yhag[0:9], label='Hagan implied vol')

plt.legend(loc="upper left")
plt.xlabel("K (Strike)")
plt.ylabel("Implied Vol")


#=== Forward Artificial Neural Network (ANN) training ===

X1=df[['rho','xi','T','v0','F0']].values  #inputs & outputs
y=df[['vol1','vol2','vol3','vol4','vol5','vol6','vol7','vol8','vol9']].values 

# Split data sets to 80% training data 20% testing data
X_train,X_test,y_train,y_test = train_test_split(X1,y,test_size=0.2,random_state=101)

Y1_train=y_train[:,[0]]	#individual outputs of training data
Y2_train=y_train[:,[1]]
Y3_train=y_train[:,[2]]
Y4_train=y_train[:,[3]]
Y5_train=y_train[:,[4]]
Y6_train=y_train[:,[5]]
Y7_train=y_train[:,[6]]
Y8_train=y_train[:,[7]]
Y9_train=y_train[:,[8]]

Y1_test=y_test[:,[0]] 	#individual outputs of testing data
Y2_test=y_test[:,[1]]
Y3_test=y_test[:,[2]]
Y4_test=y_test[:,[3]]
Y5_test=y_test[:,[4]]
Y6_test=y_test[:,[5]]
Y7_test=y_test[:,[6]]
Y8_test=y_test[:,[7]]
Y9_test=y_test[:,[8]]

# Configure TensorFlow Keras
tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name="Adam",
)

# Setup the Forward ANN
inputs=Input(shape=(5,),name='input')	# 5 inputs
x=Dense(160,activation='relu',name='160L1')(inputs)
x=Dropout(0.2)(x)				# with dropout rate
x=Dense(160,activation='relu',name='160L2')(x)
output1=Dense(1,name='vol1_out')(x)	# 9 outputs
output2=Dense(1,name='vol2_out')(x)
output3=Dense(1,name='vol3_out')(x)
output4=Dense(1,name='vol4_out')(x)
output5=Dense(1,name='vol5_out')(x)
output6=Dense(1,name='vol6_out')(x)
output7=Dense(1,name='vol7_out')(x)
output8=Dense(1,name='vol8_out')(x)
output9=Dense(1,name='vol9_out')(x)
model = Model(inputs=inputs,outputs=[output1,output2,output3,output4,output5, output6,output7,output8,output9])
model.compile(loss={'vol1_out':'mean_squared_error',
                    'vol2_out': 'mean_squared_error',
                    'vol3_out': 'mean_squared_error',
                    'vol4_out': 'mean_squared_error',
                    'vol5_out': 'mean_squared_error',
                    'vol6_out': 'mean_squared_error',
                    'vol7_out': 'mean_squared_error',
                    'vol8_out': 'mean_squared_error',
                    'vol9_out': 'mean_squared_error'
                    }, optimizer='adam')	# MSE Lose function & Adam optimizer
# Training
history=model.fit(X_train,{'vol1_out':Y1_train,'vol2_out':Y2_train,'vol3_out':Y3_train,'vol4_out':Y4_train,'vol5_out':Y5_train,'vol6_out':Y6_train,'vol7_out':Y7_train,'vol8_out':Y8_train,'vol9_out':Y9_train},batch_size=32,epochs=250)

loss_df = pd.DataFrame(history.history)  # Training history Loss
loss_df['vol1_out_loss'].plot() # Plot individual output loss
loss_df['vol2_out_loss'].plot()
loss_df['vol3_out_loss'].plot()
loss_df['vol4_out_loss'].plot()
loss_df['vol5_out_loss'].plot()
loss_df['vol6_out_loss'].plot()
loss_df['vol7_out_loss'].plot()
loss_df['vol8_out_loss'].plot()
loss_df['vol9_out_loss'].plot()
loss_df['loss'].plot()  # Plot total output loss 
plt.xlabel("Epoch")
plt.ylabel("MSE loss")

test_predictions=model.predict(X_test) # Predict testing data

explained_variance_score(Y1_test,test_predictions[0]) # variance regression score
explained_variance_score(Y2_test,test_predictions[1])
explained_variance_score(Y3_test,test_predictions[2])
explained_variance_score(Y4_test,test_predictions[3])
explained_variance_score(Y5_test,test_predictions[4])
explained_variance_score(Y6_test,test_predictions[5])
explained_variance_score(Y7_test,test_predictions[6])
explained_variance_score(Y8_test,test_predictions[7])
explained_variance_score(Y9_test,test_predictions[8])

%matplotlib qt   # Configure separate plot window
plt.plot(Y1_test,label='test output')  # Plot each predicted output with test data
plt.plot(test_predictions[0],label='predicted output')
plt.legend(loc="upper left")
plt.plot(Y2_test,label='test output')
plt.plot(test_predictions[1],label='predicted output')
plt.legend(loc="upper left")
plt.plot(Y3_test,label='test output')
plt.plot(test_predictions[2],label='predicted output')
plt.legend(loc="upper left")
plt.plot(Y4_test,label='test output')
plt.plot(test_predictions[3],label='predicted output')
plt.legend(loc="upper left")
plt.plot(Y5_test,label='test output')
plt.plot(test_predictions[4],label='predicted output')
plt.legend(loc="upper left")
plt.plot(Y6_test,label='test output')
plt.plot(test_predictions[5],label='predicted output')
plt.legend(loc="upper left")
plt.plot(Y7_test,label='test output')
plt.plot(test_predictions[6],label='predicted output')
plt.legend(loc="upper left")
plt.plot(Y8_test,label='test output')
plt.plot(test_predictions[7],label='predicted output')
plt.legend(loc="upper left")
plt.plot(Y9_test,label='test output')
plt.plot(test_predictions[8],label='predicted output')
plt.legend(loc="upper left")


#=== K-fold cross validation for forward ANN ===

allL=[i for i in range(len(X1))] # Hold indices
allL2=allL
ncv=5
partL=[[]*1 for i in range(ncv)] # To store part element indices
for j in range((ncv-1)): 		# Generate 5 parts through random sampling
    partL[j]=random.sample(allL2,int(len(X1)/5))
    allL2=[i for i in allL2 if i not in partL[j]]
partL[ncv-1]=allL2  # last part
random.shuffle(partL[ncv-1])

cvscores=[]  # Store cross validation Scores
for i in range(ncv):
    test=partL[i]  		# 1 of the 5 parts become testing data
    train=[id for id in allL if id not in test]  # remaining becomes training data
    random.shuffle(train)
    inputs=Input(shape=(5,),name='input')
    x=Dense(160,activation='relu',name='160L1')(inputs)
    x=Dropout(0.2)(x)
    x=Dense(160,activation='relu',name=’160L2')(x)
    output1=Dense(1,name='vol1_out')(x)
    output2=Dense(1,name='vol2_out')(x)
    output3=Dense(1,name='vol3_out')(x)
    output4=Dense(1,name='vol4_out')(x)
    output5=Dense(1,name='vol5_out')(x)
    output6=Dense(1,name='vol6_out')(x)
    output7=Dense(1,name='vol7_out')(x)
    output8=Dense(1,name='vol8_out')(x)
    output9=Dense(1,name='vol9_out')(x)
model2 = Model(inputs=inputs,outputs=[output1,output2,output3,output4,output5,
output6,output7,output8,output9])
    model2.compile(loss={'vol1_out':'mean_squared_error',
                        'vol2_out': 'mean_squared_error',
                        'vol3_out': 'mean_squared_error',
                        'vol4_out': 'mean_squared_error',
                        'vol5_out': 'mean_squared_error',
                        'vol6_out': 'mean_squared_error',
                        'vol7_out': 'mean_squared_error',
                        'vol8_out': 'mean_squared_error',
                        'vol9_out': 'mean_squared_error'
                        }, optimizer='adam') 
    history2=model2.fit(X1[train],{'vol1_out':y[train],'vol2_out':y[train],'vol3_out':y[train],'vol4_out':y[train],'vol5_out':y[train],'vol6_out':y[train],'vol7_out':y[train],'vol8_out':y[train],'vol9_out':y[train]},batch_size=32,epochs=250,verbose=0)
    scores=model2.evaluate(X1[test],{'vol1_out':y[test],'vol2_out':y[test],'vol3_out':y[test],'vol4_out':y[test],'vol5_out':y[test],'vol6_out':y[test],'vol7_out':y[test],'vol8_out':y[test],'vol9_out':y[test]},verbose=0)
print("%s: %.2f%%" % (model2.metrics_names[0],scores[0]*100))

cvscores.append(scores[0]*100)  # Append scores

print("%.2f%% (+/- %.2f%%)"% (np.mean(cvscores),np.std(cvscores))) # print scores

loss_df2 = pd.DataFrame(history2.history)  # Training history Loss
loss_df2['loss'].plot()			# Plot total output loss

test_predictions_2=model2.predict(X1[test]) # Predict testing data
explained_variance_score(y[test,[1]],test_predictions_2[1]) # var regression score

%matplotlib qt   # separate plot window
plt.plot(y[test,[1]],label='test output')  # Plot predicted output with test data
plt.plot(test_predictions_2[1],label='predicted output')
plt.legend(loc="upper left")


#=== Inverse Artificial Neural Network (ANN) training ===

Xinv1=df[['vol3','vol4','vol5','vol6','vol7', 'vol9','T','F0']].values #inputs
yinv=df[['rho','xi','v0']].values  #outputs

# Split data sets to 80% training data 20% testing data
Xinv_train,Xinv_test,yinv_train,yinv_test = train_test_split(Xinv1,yinv,test_size=0.2,random_state=101)

Yinv1_train=yinv_train[:,[0]]   #individual outputs of training data
Yinv2_train=yinv_train[:,[1]]
Yinv3_train=yinv_train[:,[2]]

Yinv1_test=yinv_test[:,[0]]     #individual outputs of testing data
Yinv2_test=yinv_test[:,[1]]
Yinv3_test=yinv_test[:,[2]]

# Setup the Inverse ANN
inputsinv=Input(shape=(8,),name='inputinv')   # 8 inputs
xinv=Dense(160,activation='relu',name='160L1')(inputsinv)
xinv = Dropout(0.2)(xinv) 				# with dropout rate
xinv=Dense(160,activation='relu',name='160L2')(xinv)
outputinv1=Dense(1,name='rho_out')(xinv)	# 3 outputs
outputinv2=Dense(1,name='xi_out')(xinv)
outputinv3=Dense(1,name='v0_out')(xinv)
modelinv = Model(inputs=inputsinv,outputs=[outputinv1,outputinv2,outputinv3])
modelinv.compile(loss={'rho_out':'mean_squared_error',
                       'xi_out': 'mean_squared_error',
                       'v0_out':'mean_squared_error'},
                    optimizer='adam')  # MSE Lose function & Adam optimizer
# Training
historyinv=modelinv.fit(Xinv_train,{'rho_out':Yinv1_train,'xi_out':Yinv2_train,'v0_out':Yinv3_train},batch_size=32, epochs=250)


lossinv_df = pd.DataFrame(historyinv.history)  # Training history Loss
lossinv_df['rho_out_loss'].plot()  # Plot individual output loss
lossinv_df['xi_out_loss'].plot()
lossinv_df['v0_out_loss'].plot()
lossinv_df['loss'].plot()    # Plot total output loss
plt.xlabel("Epoch")
plt.ylabel("MSE loss")

test_predictions_inv=modelinv.predict(Xinv_test)  # Predict testing data

%matplotlib qt   # Separate Plot Window
plt.plot(Yinv1_test,label='test output')   # Plot predicted output with test data
plt.plot(test_predictions_inv[0],label='predicted output')
plt.legend(loc="upper left")
plt.plot(Yinv2_test,label='test output')   
plt.plot(test_predictions_inv[1],label='predicted output')
plt.legend(loc="upper left")
plt.plot(Yinv3_test,label='test output')   plt.plot(test_predictions_inv[2],label='predicted output')
plt.legend(loc="upper left")

explained_variance_score(Yinv1_test,test_predictions_inv[0]) # Variance regression score
explained_variance_score(Yinv2_test,test_predictions_inv[1])
explained_variance_score(Yinv3_test,test_predictions_inv[2])


#=== K-fold cross validation for inverse ANN ===

allL=[i for i in range(len(Xinv1))]  	# Hold indices
allL2=allL
ncv=5
partL=[[]*1 for i in range(ncv)]		# To store part element indices
for j in range((ncv-1)): 			# Generate 5 parts through random sampling
    partL[j]=random.sample(allL2,int(len(Xinv1)/5))
    allL2=[i for i in allL2 if i not in partL[j]]
partL[ncv-1]=allL2  # last part
random.shuffle(partL[ncv-1])

cvscores=[]	# Store cross validation Scores
for i in range(ncv):
    test=partL[i]		# 1 of the 5 parts become testing data
    train=[id for id in allL if id not in test]  # remaining becomes training data
    random.shuffle(train)
    inputsinv2=Input(shape=(8,),name='inputinv')
    xinv2=Dense(160,activation='relu',name='160L1')(inputsinv2)
    xinv2 = Dropout(0.2)(xinv2)
    xinv2=Dense(160,activation='relu',name='160L2')(xinv2)
    outputinv1b=Dense(1,name='rho_out')(xinv2)
    outputinv2b=Dense(1,name='xi_out')(xinv2)
    outputinv3b=Dense(1,name='v0_out')(xinv2)
    modelinv2 = Model(inputs=inputsinv2,outputs=[outputinv1b,outputinv2b, outputinv3b])
    modelinv2.compile(loss={'rho_out':'mean_squared_error',
                           'xi_out': 'mean_squared_error',
                           'v0_out':'mean_squared_error'},
                        optimizer='adam')

historyinv2=modelinv2.fit(Xinv1[train],{'rho_out':yinv[train,[0]],'xi_out':yinv[train,[1]],'v0_out':yinv[train,[2]]},verbose=0,epochs=250)
    scores=modelinv2.evaluate(Xinv1[test],{'rho_out':yinv[test,[0]],'xi_out':yinv[test,[1]],'v0_out':yinv[test,[2]]},verbose=0)
print("%s: %.2f%%" % (modelinv2.metrics_names[0],scores[0]*100))

cvscores.append(scores[0]*100)  # Append Scores

print("%.2f%% (+/- %.2f%%)"% (np.mean(cvscores),np.std(cvscores))) # print scores

lossinv_df2 = pd.DataFrame(historyinv2.history)  # Training history Loss
lossinv_df2['loss'].plot()				# Plot total output loss

test_predictions_inv2=modelinv2.predict(Xinv1[test])  # Predict testing data
explained_variance_score(yinv[test,[1]],test_predictions_inv2[1]) #var regression score 

%matplotlib qt   # Separate Plot Window
plt.plot(yinv[test,[1]],label='test output') # Plot predicted output with test data
plt.plot(test_predictions_inv2[1],label='predicted output')
plt.legend(loc="upper left")


#=== Input Market Data ===

# Read market imp volatility from CSV file (of different tenors 2Y,4Y,...)
mktdatraw=[]  # Hold raw market data
with open(r'C:\Users\Documents\Python files\swap_USD_tenor2yr.csv', 'r',) as fileinp:
    reader = csv.reader(fileinp)
    for rowm in reader:
        datam=rowm
        mktdatraw.append(np.array(datam))

mktdat=[['']*1 for s in range(len(mktdatraw)-1)] # Hold market data without header
for n in range(1,len(mktdatraw)):
    mktdat[n-1]=mktdatraw[n]
    for m in range(len(mktdatraw[0])-2):  		 
        mktdat[n-1][m]=float(mktdat[n-1][m])/100	# convert % to decimal

mktinp=[[]*1 for s in range(len(mktdat))]	# Hold array of list of market data
for n in range(len(mktdat)):		# Loop each T
    mktinp[n]=mktdat[n].tolist()			# Convert into list
    for m in range(len(mktinp[0])):
        mktinp[n][m]=float(mktinp[n][m])

pk1=[]   # store inverse ANN inferred rho
pk2=[]   # store inverse ANN inferred xi
pk3=[]   # store inverse ANN inferred v0 
F0n=[]
for n in range(len(mktinp)):		# Loop each T
    mktpred=modelinv.predict([mktinp[n]])  # Inverse ANN predictions
    pk1.append(float(mktpred[0][0]))	# Append rho
    pk2.append(float(mktpred[1][0]))	# Append xi
    pk3.append(float(mktpred[2][0]))	# Append v0
    F0n.append(mktinp[n][len(mktinp[n])-1])	# Append F0


#=== Calibration (Find Optimized Parameters) ===

# Prepare list of inverse ANN parameters for different T
input_data=pd.DataFrame([[pk1[0],pk2[0],0.5,pk3[0],F0n[0]],[pk1[1],pk2[1],1,pk3[1],F0n[1]],[pk1[2],pk2[2],2,pk3[2],F0n[2]],[pk1[3],pk2[3],5,pk3[3],F0n[3]],[pk1[4],pk2[4],10,pk3[4],F0n[4]],[pk1[5],pk2[5],20,pk3[5],F0n[5]]])

x_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32) #conv. to tensor obj.

# Configure TensorFlow Automatic Differentiation for gradient computation
with tf.GradientTape(persistent=True) as tape:
    tape.watch(x_tensor)
    output = model(x_tensor)
    
grad=[]  # Hold gradients
grad.append(tape.gradient(output[2], x_tensor))  # gradient of f-ANN output 2 
grad.append(tape.gradient(output[3], x_tensor))  # gradient of f-ANN output 3
grad.append(tape.gradient(output[4], x_tensor))  # gradient of f-ANN output 4
grad.append(tape.gradient(output[5], x_tensor))  # gradient of f-ANN output 5
grad.append(tape.gradient(output[6], x_tensor))  # gradient of f-ANN output 6
grad.append(tape.gradient(output[8], x_tensor))  # gradient of f-ANN output 8

# rho
prod_sum1=0
weight_sum1=0
weight1=0
for i in range(len(mktinp)):  # no. of T
    for j in range(6):   # no. of impvol of different K
        weight1=(1/(6*len(mktinp)))*abs(grad[j][i].numpy()[0])
        prod_sum1+=weight1*pk1[i]
        weight_sum1+=weight1
rho_opt=prod_sum1/weight_sum1	# Calibrated/Optimum Rho

# xi
prod_sum2=0
weight_sum2=0
weight2=0
for i in range(len(mktinp)):  # no. of T
    for j in range(6):   # no. of impvol of different K
        weight2=(1/(6* len(mktinp)))*abs(grad[j][i].numpy()[1])
        prod_sum2+=weight2*pk2[i]
        weight_sum2+=weight2
xi_opt=prod_sum2/weight_sum2	# Calibrated/Optimum Xi

# V0
prod_sum3=0
weight_sum3=0
weight3=0
for i in range(len(mktinp)):  # no. of T
    for j in range(6):   # no. of impvol of different K
        weight3=(1/(6*len(mktinp)))*abs(grad[j][i].numpy()[3])
        prod_sum3+=weight3*pk3[i]
        weight_sum3+=weight3
v0_opt=prod_sum3/weight_sum3	# Calibrated/Optimum V0


#=== Calibration (Check Performance) ===

# Calculate RMSE with market data based on Hagan's approximation
rho_=rho_opt
xi_=xi_opt
v0_=v0_opt
#rho_=-0.4874  # Excel Solver’s Hagan’s approx. optimized parameters
#xi_=1
#v0_=0.6279 
pdif_sum=0
voldat=[[]*1 for s in range(len(mktinp))]
perdif=[[]*1 for s in range(len(mktinp))]
results=[[]*1 for s in range(len(mktinp))]  # Holds Hagan’s implied vol.
off_list=mktdatraw[0][0:-2]
for n in range(len(mktinp)):	# Loop each T
    Tn_=mktinp[n][len(mktinp[n])-2]
    F0n_=mktinp[n][len(mktinp[n])-1]
    for m in range(len(off_list)):
        Km_=F0n_+0.01*float(off_list[m])  # Relative Strikes
        impvol=calc_impvol(rho_,xi_,v0_,Tn_,F0n_,Km_,beta)  # Hagan’s imp. vol.
        results[n].append(impvol)
        print("%.4f vs %.4f"%(impvol,mktinp[n][m]))
        pdif=(impvol-mktinp[n][m])**2	# Square Error
        pdif_sum+=pdif			# Sum
        voldat[n].append(impvol)
        perdif[n].append(pdif)
    print('\n')
print(np.sqrt(pdif_sum/(6*len(mktinp))))  # Print RMSE

# Eg. mktinp[0]= [1.8037, 0.9045, 0.7598, 0.7461, 0.7593, 0.7937, 0.5, 1.13]

# Calculate RMSE with market data based on ANN prediction
rho_=rho_opt
xi_=xi_opt
v0_=v0_opt
#rho_=-0.4874 # Excel Solver’s Hagan’s approx. optimized parameters
#xi_=1 
#v0_=0.6279 
pdif_sum2=0
voldat2=[[]*1 for s in range(len(mktinp))]
perdif2=[[]*1 for s in range(len(mktinp))]
results2=[[]*1 for s in range(len(mktinp))]  # Holds forward ANN implied vol.
for n in range(len(mktinp)):			# Loop each T
    Tn_=mktinp[n][len(mktinp[n])-2]
    F0n_=mktinp[n][len(mktinp[n])-1]
    opt_inp=pd.DataFrame([[rho_,xi_,Tn_,v0_,F0n_]])  # Optimized param inputs
    vol_pred=model.predict(opt_inp)	# Forward ANN prediction based on param
    results2[n].append(float(vol_pred[2]))
    print("%.4f vs %.4f"%(float(vol_pred[2]),mktinp[n][0]))
    pdif2=(float(vol_pred[2])-mktinp[n][0])**2		# Square Error
    perdif2[n].append(pdif2)
    pdif_sum2+=pdif2					# sum
    results2[n].append(float(vol_pred[3]))
    print("%.4f vs %.4f"%(float(vol_pred[3]),mktinp[n][1]))
    pdif2=(float(vol_pred[3])-mktinp[n][1])**2		# Square Error
    perdif2[n].append(pdif2)
    pdif_sum2+=pdif2					# sum
    results2[n].append(float(vol_pred[4]))
    print("%.4f vs %.4f"%(float(vol_pred[4]),mktinp[n][2]))
    pdif2=(float(vol_pred[4])-mktinp[n][2])**2		# Square Error
    perdif2[n].append(pdif2)
    pdif_sum2+=pdif2					# sum
    results2[n].append(float(vol_pred[5]))
    print("%.4f vs %.4f"%(float(vol_pred[5]),mktinp[n][3]))
    pdif2=(float(vol_pred[5])-mktinp[n][3])**2		# Square Error
    perdif2[n].append(pdif2)
    pdif_sum2+=pdif2					# sum
    results2[n].append(float(vol_pred[6]))
    print("%.4f vs %.4f"%(float(vol_pred[6]),mktinp[n][4]))
    pdif2=(float(vol_pred[6])-mktinp[n][4])**2		# Square Error
    perdif2[n].append(pdif2)
    pdif_sum2+=pdif2					# sum
    results2[n].append(float(vol_pred[8]))
    print("%.4f vs %.4f"%(float(vol_pred[8]),mktinp[n][5]))
    pdif2=(float(vol_pred[8])-mktinp[n][5])**2		# Square Error
    perdif2[n].append(pdif2)
    pdif_sum2+=pdif2					# sum
    print('\n')
print(np.sqrt(pdif_sum2/(6*len(mktinp)))) 	# Print RMSE

#=== Plot 3D surface ===

off_list2=[]  # Holds Relative strikes w.r.t ATM
for m in range(len(mktdatraw[0])-2):
        off_list2.append(float(mktdatraw[0][m])/100)

mktq=[[]*1 for s in range(len(mktinp))]	# Holds Market quotes (exclude T, Fo)
for n in range(len(mktinp)):
    mktq[n]=mktinp[n][0:-2]
    
X=off_list2
Y=Tlist
X,Y = np.meshgrid(X,Y)
Z1=np.array(mktq)
Z2=np.array(results2)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})  # Separate 3D plot
ax.view_init(0, -50)   # Adjust 3D view angle
surf1=ax.plot_surface(X,Y,Z1,cmap=cm.winter,linewidth=0,antialiased=False,label='market imp vol')
surf2=ax.plot_surface(X,Y,Z2,cmap=cm.summer,linewidth=0,antialiased=False,label='ANN calib imp vol')
fig.colorbar(surf2,shrink=0.5,aspect=5) # plot surface
fig.colorbar(surf1,shrink=0.5,aspect=5) # plot surface
ax.set_xlabel('relative K (strike)')
ax.set_ylabel('T (maturity)')
ax.legend()    
