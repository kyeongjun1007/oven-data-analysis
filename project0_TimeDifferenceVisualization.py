import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import scipy.stats as stats
%matplotlib inline

from tqdm.notebook import tqdm

os.chdir("C:/Users/USER/Desktop/LG")

"""------------------------------------------------------------------------------------------------------
### 1. comparing setted cooktime and recorded localtime
------------------------------------------------------------------------------------------------------"""

df=pd.read_csv('Oven_sample.csv',encoding='cp949',parse_dates=['CREATE_DT','LOCAL_TIME','MONTH'],dayfirst=True)
df=df.drop(['LOCAL_TIME','Target_temp','MONTH'],axis=1) # 변수 선택
df['time']=df['Cook_hour'].astype(int)*60*60 + df['Cook_min'].astype(int)*60 + df['Cook_sec'].astype(int) # 요리 시간 변수 생성

#id_00000인 제품으로 시각화
df1=df[df['DEVICE_ID']=='id_00000']

df1=df1.sort_values(by='CREATE_DT',ascending=True).reset_index(drop=True) #시간순으로 정렬

## 1-1. 시간 차이 계산
df1['CREATE_diff']=pd.Series()

for i in tqdm(range(len(df1))):
    if i != len(df1)-1:
        df1.loc[i,'CREATE_diff']=(df1['CREATE_DT'][i+1]-df1['CREATE_DT'][i]).seconds
    else:
        df1.loc[i,'CREATE_diff']=0
        
## 1-2. 요리가 시작/재시작 이후 멈춤 혹은 종료한 경우만 추출
li=[]
for i in tqdm(range(len(df1))):
    if i != len(df1)-1:
        if ((df1.loc[i,'EVENT'] in ['요리시작','요리재시작']) & (df1.loc[i+1,'EVENT'] in ['요리멈춤','요리종료','요리취소'])) | ((df1.loc[i,'EVENT']=='청소시작') & (df1.loc[i+1,'EVENT']=="청소종료")):
            li.append(i)
            li.append(i+1)

df2=df1.loc[li,]

## 1-3. 시각화
df3=df2.reset_index(drop=True)
df3=df3.ix[::2,:]
x=df3['time']
y=df3['CREATE_diff']

plt.scatter(x,y)
plt.plot(y,y)

y[y>20000] # 이상치 때문에 판단하기 어려움이 있음

## 1-4. 이상치 제거후 다시 시각화
df3=df3.drop([180,810,962,1996],axis=0)

x=df3['time']
y=df3['CREATE_diff']

plt.scatter(x,y)
plt.plot(y,y,color='red')