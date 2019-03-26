import random
from sklearn.kernel_ridge import KernelRidge
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import optuna

data=np.genfromtxt("data.csv",delimiter=",",dtype="float")

X1=data[0:"",0:""]
Y=data[0:"","":""]

(x_train, x_test,
y_train, y_test) = train_test_split(
X1, Y, test_size=0.2, random_state=0,
)
pnum=Y.shape[1]


best_score=np.zeros(pnum)
best_gamma=np.zeros(pnum)
best_alpha=np.zeros(pnum)


for i in range(0,pnum):
    print(i)
    for gamma1 in range(-18,18):
        for alpha1 in range(-18,18):
           gamma=10**(gamma1)
           alpha=10**(alpha1)
           ker = KernelRidge(alpha=alpha,gamma=gamma,kernel='rbf')
           ker.fit(x_train,y_train[:,i])

           score=cross_val_score(ker,x_train,y_train[:,i],cv=5)
           score=np.mean(score)

           print(gamma,alpha,score)
           if score > best_score[i]:
              best_score[i]=score
              best_gamma[i]=gamma
              best_alpha[i]=alpha
              best_coef=ker.dual_coef_
        print("")

    if i==0:
        coef_merge=best_coef
    if i>0:
        coef_merge = np.c_[coef_merge, best_coef]

    best_parameters={'alpha':best_alpha[i],'gamma':best_gamma[i]}
    print("BestScore:{:.2f}".format(best_score[i]))
    print("Best parameters:{}".format(best_parameters))

df=pd.DataFrame(coef_merge)
df.to_csv("best_coef.csv",header=False,index=False,)
df=pd.DataFrame(best_gamma)
df.to_csv("best_gamma.csv",header=False,index=False,)
df=pd.DataFrame(best_alpha)
df.to_csv("best_alpha.csv",header=False,index=False,)

for j in range(0,pnum):

   clf = KernelRidge(alpha=best_alpha[j], kernel='rbf',gamma=best_gamma[j])
   clf.fit(x_train,y_train[:,j])
       #print(clf.dual_coef_)

   predict = clf.predict(x_test)
if j==0:
   predict_merge=predict
if j>0:
   predict_merge = np.c_[predict_merge, predict]


print("------------------------------------------------input result VAL-------------------------------------------------------------------------")


for j in range(0,pnum):
   clf = KernelRidge(alpha=best_alpha[j], kernel='rbf',gamma=best_gamma[j])
   clf.fit(x_train,y_train[:,j])
    #print(clf.dual_coef_)

   predict = clf.predict(x_test)
   print("R2:",r2_score(y_test[:,j],predict)," ","MAE:",mean_absolute_error(y_test[:,j],predict))
if j==0:
   predict_merge=predict
if j>0:
   predict_merge = np.c_[predict_merge, predict]

print("------------------------------------------------input result PRED-------------------------------------------------------------------")



print("-------------------------START RANDOM SERACH to FIND PARAMS!!-----------------------------------------")
trials=1000
for i in range(0,trials):
    para0=random.randrange(30, 70, 1)
    para1=random.randrange(30, 70, 1)
    para2=random.randrange(30, 70, 1)
    para3=random.randrange(30, 70, 1)
    para4=random.randrange(30, 70, 1)
    paramlist0=np.append(para0,para1)
    paramlist1=np.append(paramlist0,para2)
    paramlist2=np.append(paramlist1,para3)
    paramlist3=np.append(paramlist2,para4)
    paramlist=paramlist3.reshape(1,-1) 
    predict = clf.predict(paramlist)
    params=np.append(paramlist,predict)
    df=pd.DataFrame([params])
    df.to_csv("random_params.csv",mode='a',header=False,index=False,)

print("-----------------FIN RANDOM SERACH to FIND PARAMS!!----------------------")
