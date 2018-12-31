
# coding: utf-8

# In[9]:


# Importing libraries
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score



# In[31]:


#Importing Data
dataTrain = pd.read_csv("DataTrain_Tugas3_AI.csv")
#dataTrain.head()
#print(dataTrain)
dataTest = pd.read_csv("DataTest_Tugas3_AI.csv")
#dataTest.head()
#print(dataTest)


# In[52]:



#datavalidation
    
#0
filtery0= dataTrain['Y'] == 0
filter0 =dataTrain[filtery0]
fily0=filter0[:25]

#1
filtery1= dataTrain['Y'] == 1
filter1 =dataTrain[filtery1]
fily1=filter1[:25]

#2

filtery2= dataTrain['Y'] == 2
filter2 =dataTrain[filtery2]
fily2=filter2[:25]
#3

filtery3= dataTrain['Y'] == 3
filter3 =dataTrain[filtery3]
fily3=filter3[:25]
#

frames =[fily0,fily1,fily2,fily3]

dataValidation = pd.concat(frames)

dataTrains= dataTrain.drop(dataValidation.index.values)

yval=dataValidation['Y']

#Penentuan Jarak


def hitungjarakpakedata(xtr,xts):
    jarak=0
    for prm in range(len(xtr)-1):
        if prm>0:
            jarak += np.square(xtr[prm]-xts[prm])
    return np.sqrt(jarak)

#cari k optimum

ypred=[]
k=15
#

#for i in range(len(dataValidation)):
#    z=[]
#  
#    for j in range(len(dataTrains)):
#        a=hitungjarakpakedata(dataTrains.iloc[j],dataValidation.iloc[i])
#        z.append(a)
#    srt=np.argsort(z)[:k]
#    mod= stats.mode(np.array((dataTrains.iloc[srt]['Y'])))[0]
#    ypred.append(mod)
#    
#acc=accuracy_score(yval,ypred)
#    pred.append(acc)

#print('akurasi dengan :', len(dataValidation),'data, menghasilkan akurasi sebesar',acc*100,' dengan k =',k)

ypredict=[]

#knn
for i in range(len(dataTest)):
    z=[]
    a=[]
   
    for j in range(len(dataTrain)):
        a=hitungjarakpakedata(dataTrain.iloc[j],dataTest.iloc[i])
        z.append(a)
    srt=np.argsort(z)[:k]
    modus= stats.mode(np.array((dataTrain.iloc[srt]['Y'])))[0]
    ypredict.append(modus)



fix= pd.DataFrame(ypredict)
fix.to_csv('TebakanTugas3.csv',index=False,header=False)
print(fix)