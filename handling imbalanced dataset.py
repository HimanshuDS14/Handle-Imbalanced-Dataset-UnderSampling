#if you have millions of record in your dataset than you can apply undersampling algorithm



import pandas as pd
import numpy as np

from imblearn.under_sampling import NearMiss

data = pd.read_csv("creditcard_csv.csv")
#print(data.head(10))

#print(data.info())

x = data.iloc[:,:30].values
y = data.iloc[:,30].values
#print(x)
#print(y)


count = data["Class"].value_counts()

print("Before sampling Fraud transaction is :  " , count[0])
print("Before sampling normal transaction is : " , count[1])

#this data set is totally unbalanced because normal transaction is actually greater than fraud transaction

nm = NearMiss()
x_res , y_res = nm.fit_sample(x,y)
print("*******************NearMiss Algorithm************************")
print(x_res.shape)
print(y_res.shape)

fraud = 0
normal = 0

for i in y_res:
    if i=="'1'":
        fraud = fraud+1
    else:
        normal = normal+1
print("After sampling Fraud Tramsaction is : " , fraud)
print("After sampling Normal Transaction is : " , normal)
