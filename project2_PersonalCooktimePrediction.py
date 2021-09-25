import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
import numpy as np

os.chdir("C:/Users/Kyeongjun/Desktop/oven")

"""------------------------------------------------------------------------------------------------------
### 1. calculating expected cooktime of 27 menus
------------------------------------------------------------------------------------------------------"""
## 1-1. 27개 메뉴 선택 & 가나다순 정렬
# 메뉴 선택 기준 : 100회 이상 요리된 메뉴 &10명 이상의 user(device)가 요리한 메뉴
oven = pd.read_csv('Oven_cooktimePred.csv', encoding='euc-kr')

menu = list(set(oven.Cook_menu)) # 27개
menu.sort()

## 1-2. menu별 expected cooktime 계산(mean, mode, median)
name, mean, mode, median = [], [], [], []

for i in range(27):
    df = oven.loc[oven['Cook_menu'] == menu[i],['Cook_menu', 'Cookingtime']]
    df1 = pd.DataFrame(df.Cookingtime.value_counts()).sort_index()  
    df1 = pd.DataFrame(list(zip(list(df1.index),list(df1.Cookingtime))), columns= ['time', 'count'])
    
    m1 = round(sum(df1.iloc[:,0]*df1.iloc[:,1])/sum(df1.iloc[:,1]), 2)
    m2 = df1.sort_values(by='count').iloc[len(df1)-1,0]
    k = sum(df1['count'])/2
    o = (k%2 == 0)
    df0 = df.sort_values(by='Cookingtime')
    df0 = df0.reset_index(drop=True)
    if o :
        m3 = (df0['Cookingtime'][round(k)] + df0['Cookingtime'][round(k)+1])/2
    else :
        m3 = df0['Cookingtime'][round(k)]
        
    name.append(menu[i]); mean.append(m1); mode.append(m2); median.append(m3)

ex_cooktime = pd.DataFrame(list(zip(name,mean,mode,median)), columns = ['menu','mean','mode','median'])
del m1, m2, m3, df, df1, df0, k, o, mean, mode, median, name, i

ex_cooktime.to_csv('ex_cooktime.csv', header=True, index=False, encoding = 'euc-kr')

"""------------------------------------------------------------------------------------------------------
### 2. NN (user charicteristics + menu)
------------------------------------------------------------------------------------------------------"""
## 2-1. cooktime prediction용 DF 생성
oven_c = pd.read_csv('oven_clustering.csv')         # user charicteristics df

oven = pd.read_csv('Oven_sample.csv', encoding = 'euc-kr')
oven['Cookingtime'] = oven['Cook_hour']*60*60 + oven['Cook_min']*60 + oven['Cook_sec']  # Cookingtime변수 생성 (시,분,초를 합산)
oven = oven.loc[oven['EVENT'] == '요리시작',:]                                          # 요리 시작에 관한 데이터만 추출
oven = oven.loc[oven['Cook_menu'].isin(menu),]                                          # 27개 메뉴에 관한 데이터만 추출
oven = oven.iloc[:,[2,6,12]]                                                            # DVICE_ID, Cook_menu, Cookingtime 변수만 선택

ctPred1 = pd.merge(oven_c,oven, left_on = 'DEVICE_ID', right_on = 'DEVICE_ID', how = 'inner')

ctPred1.columns = list(ctPred1.columns[:23]) +['Y']

# NN에 입력하기 위해 string으로 표현된 Cook_menu를 각각 0~26의 숫자로 대체합니다.
for i in range(len(ctPred1)) :
    k = ctPred1.iloc[i,22]
    num = menu.index(k)
    ctPred1.iloc[i,22] = num
    
del k, num, i

ctPred1 = ctPred1.iloc[:,1:]

ctPred1.to_csv('ctPred1.csv', header=True, index=False)

del oven
## 2-2. NeuralNetwork modeling
# train_test_split
train, test = train_test_split(ctPred1, test_size = 0.2)

## make NN model
NN = keras.Sequential()
NN.add(keras.layers.Input(shape = (train.shape[1]-1)))  # 종속변수 Y를 제외한 변수 개수를 input size로 설정
NN.add(keras.layers.Dense(11, activation = "relu"))     # hidden layer의 node 개수는 input변수의 개수와 output 변수의 개수의 중간 값으로 설정
NN.add(keras.layers.Dense(1, activation = None))        # output변수. 연속형이므로 activation function을 설정하지 않습니다.

Adam = keras.optimizers.Adam(learning_rate = 0.001)     # optimizer는 Adam, learning_rate는 0.001

NN.compile(optimizer = Adam, loss = 'mse', metrics = ['mse'])           # mse를 기준으로 모델을 평가/학습합니다.

# model fitting & evaluation
NN.fit(train.iloc[:,:-1],train.iloc[:,-1], epochs = 30, batch_size=10)  # batch_size = 10으로 설정하여 한번에 10개 row씩 학습됩니다.
                                                                        # epochs = 30으로 설정하여 train set을 30번 학습시킵니다.

NN.evaluate(test.iloc[:,:-1], test.iloc[:,-1])                          # test set에서의 성능을 확인입니다.

(600000**(1/2))/60 # 대략 +-12분정도의 차이를 보임

"""------------------------------------------------------------------------------------------------------
### 3. NN (user charicteristics + menu + accumulate/recent session variables)
------------------------------------------------------------------------------------------------------"""
## 3-1. 추가 변수 DF 생성
# 생성 변수 종류
# 동작 이전 : 메뉴 요리 횟수, 레인지 평균 요리 시간, 오븐 평균 요리 시간, 
# 이전 세션 : 메뉴 요리 여부, 요리 시간 
# 기타 : 현지 시간(시), 월

# 동작 이전 누적 변수 생성
oven_av = pd.read_csv('Oven_sample.csv', encoding = 'euc-kr')
oven_av['Cookingtime'] = oven_av['Cook_hour']*60*60 + oven_av['Cook_min']*60 + oven_av['Cook_sec']
oven_av = oven_av.loc[oven_av['EVENT'] == '요리시작',]
oven_av['Micro_t'], oven_av['Oven_t'], oven_av['MenuFreq'] = 0,0,0

df = pd.DataFrame()

for i in list(set(oven_av.DEVICE_ID)) :
    df0 = oven_av.loc[oven_av['DEVICE_ID'] == i,:]
    df0 = df0.sort_values(by='LOCAL_TIME')
    for j in range(df0.shape[0]) :
        df1 = df0.copy(deep=True)
        df1 = df1.iloc[:j+1,:]
        k = df1.loc[df1['Cook_Mode'].isin(['레인지', '레인지 자동']),'Cookingtime']
        if len(k) != 0 :
            df0.iloc[j,13] = sum(k)/len(k)
        k = df1.loc[df1['Cook_Mode'].isin(['오븐', '오븐 자동']),'Cookingtime']
        if len(k) != 0 :
            df0.iloc[j,14] = sum(k)/len(k)
        k = df1.loc[df1['Cook_menu'] == df0.iloc[j,6],'Cookingtime']
        df0.iloc[j,15] = len(k)
    
    df = df.append(df0)

oven_av = df.copy(deep=True)

del i, j, df0, df1, k, df

oven_av = oven_av.loc[oven_av['Cook_menu'].isin(menu),]     # 27개 메뉴 데이터 추출

# 이전 세션 변수 생성 (1hour rule)
session = pd.read_csv('session.csv', encoding='euc-kr')
session['S1'], session['S2'] = 0, 0

for i in range(session.shape[0])[1:] :
    s = session.loc[session['Session2'] == session.iloc[i,11]-1,:]
    session.iloc[i,14] = sum(s['Cook_hour'])*60*60 + sum(s['Cook_min'])*60 + sum(s['Cook_sec'])
    session.iloc[i,13] = 1 if (session.iloc[i,5] in list(s.Cook_menu)) else 0

session = session.loc[session['Cook_menu'].isin(menu),:]  # 27개 메뉴 데이터 추출

del s, i,

# 기타 변수까지 포함한 DataFrame 생성
oven_av = oven_av.sort_values(by=['DEVICE_ID','LOCAL_TIME'])        # DEVICE_ID, LOCAL_TIME으로 정렬한 뒤, 
df0 = oven_av.iloc[:,[1,2,6,12,13,14,15]]                           # 변수 선택
df0 = df0.reset_index(drop=True)                                    # session 변수와 합칠 때 index별로 concatenate 되지 않도록 index reset
session = session.sort_values(by=['DEVICE_ID','LOCAL_TIME'])        # session 데이터도 위와 동일한 작업을 합니다.
df1 = session.iloc[:,[1,13,14]]
df1 = df1.reset_index(drop=True)

df = pd.concat([df0,df1.iloc[:,[1,2]]],axis=1)                      # 동작이전 누적 변수와 이전세션 변수 통합
ctPred2 = pd.merge(oven_c,df, left_on = 'DEVICE_ID', right_on = 'DEVICE_ID', how = 'inner')    # 위 통합 데이터와 DEVICE_ID별 특징변수 통합
ctPred2 = ctPred2.sort_values(by=['DEVICE_ID','LOCAL_TIME'])        # DEVICE_ID, LOCAL_TIME으로 정렬

ctPred2.to_csv('ctPred2_preprocessing.csv', header=True, index=False, encoding='euc-kr')

del df0, df1, df

# NN에 입력하기 위해 string으로 표현된 Cook_menu를 각각 0~26의 숫자로 대체합니다.
for i in range(len(ctPred2)) :
    k = ctPred2.iloc[i,23]
    num = menu.index(k)
    ctPred2.iloc[i,23] = num
    
del k, num, i

ctPred2['time'], ctPred2['month'] = 0, 0

for i in range(len(ctPred2)) :
    ctPred2.iloc[i,30] = int(ctPred2.iloc[i,22][11:13])   # time 변수 생성
    ctPred2.iloc[i,31] = int(ctPred2.iloc[i,22][5:7])     # month 변수 생성
    
del i, ctPred2['LOCAL_TIME'], ctPred2['DEVICE_ID']

# column 순서 변경 (Cookingtime = Y 를 끝으로)
cols = ctPred2.columns.tolist()
cols = cols[:cols.index('Cookingtime')] + cols[cols.index('Cookingtime')+1:]
cols.append('Cookingtime')
ctPred2 = ctPred2[cols]

cols = ctPred2.columns.tolist()
cols[len(cols)-1] = 'Y'
ctPred2.columns = cols

del cols

ctPred2.to_csv('ctPred2.csv', header=True, index=False)

## 3-2. NeuralNetwork modeling 2
# train_test_split
train, test = train_test_split(ctPred2, test_size = 0.2)

## 3-3. make NN model
NN = keras.Sequential()
NN.add(keras.layers.Input(shape = (train.shape[1]-1)))
NN.add(keras.layers.Dense(18, activation = "relu"))     # hidden layer의 layer 개수는 2, node 개수는 input변수의 개수와 output 변수의 개수의 1/3, 1/3 지점으로 설정
NN.add(keras.layers.Dense(9, activation = "relu"))
NN.add(keras.layers.Dense(1, activation = None))

Adam = keras.optimizers.Adam(learning_rate = 0.001)

NN.compile(optimizer = Adam, loss = 'mse', metrics = ['mse'])

NN.fit(train.iloc[:,:-1],train.iloc[:,-1], epochs = 70, batch_size=1)

NN.evaluate(test.iloc[:,:-1], test.iloc[:,-1])

(400000**(1/2))/60 # 대략 +-10분정도의 차이를 보임

"""------------------------------------------------------------------------------------------------------
### 4. NN (user charicteristics + menu + accumulate/recent session variables + recent cooktime)
------------------------------------------------------------------------------------------------------"""
## 4-1. recent cooktime 변수 DF 생성
oven_rct = pd.read_csv('Oven_sample.csv', encoding='euc-kr')

oven_rct['Cookingtime'] = oven_rct['Cook_hour']*60*60 + oven_rct['Cook_min']*60 + oven_rct['Cook_sec']
oven_rct = oven_rct.loc[oven_rct['EVENT'] == '요리시작',]
oven_rct = oven_rct.loc[oven_rct['Cook_menu'].isin(menu),]


oven_rct['recent_ct'] = 0

df = pd.DataFrame()

for i in list(set(oven_rct.DEVICE_ID)) :
    df0 = oven_rct.loc[oven_rct['DEVICE_ID'] == i,:]
    df0 = df0.sort_values(by='LOCAL_TIME')
    for j in range(1,df0.shape[0]) :
        df1 = df0.copy(deep=True)
        df2 = df1.iloc[:j,:]
        k = list(df2.loc[df2['Cook_menu'] == df1.iloc[j,6],'Cookingtime']) if (df1.iloc[j,6] in list(set(df2['Cook_menu']))) else [0]
        k = k[len(k)-1]
        df0.iloc[j,13] = k
    
    df = df.append(df0)

oven_rct = df.copy(deep=True)
oven_rct = oven_rct.reset_index(drop=True)

del i, j, df0, df1, k, df, df2

df = pd.read_csv('ctPred3_preprocessing.csv', encoding='euc-kr')

df = pd.concat([df,oven_rct.iloc[:,-1]],axis=1)
ctPred3 = pd.merge(oven_c,df, left_on = 'DEVICE_ID', right_on = 'DEVICE_ID', how = 'inner')
ctPred3 = ctPred3.sort_values(by=['DEVICE_ID','LOCAL_TIME'])

# menu를 가나다순으로 0-27까지 레이블링
for i in range(len(ctPred3.iloc[:,0])) :
    k = ctPred3.iloc[i,23]
    num = menu.index(k)
    ctPred3.iloc[i,23] = num
    
del k, num, i

ctPred3['time'] = 0

for i in range(len(ctPred3.iloc[:,0])) :
    ctPred3.iloc[i,31] = int(ctPred3.iloc[i,22][11:13])   # time 변수 생성
    
del i
del ctPred3['LOCAL_TIME'], ctPred3['DEVICE_ID']

# column 순서 변경 (Cookingtime = Y 를 끝으로)
cols = ctPred3.columns.tolist()
cols = cols[:cols.index('Cookingtime')] + cols[cols.index('Cookingtime')+1:]
cols.append('Cookingtime')
ctPred3 = ctPred3[cols]

cols = ctPred3.columns.tolist()
cols[len(cols)-1] = 'Y'
ctPred3.columns = cols

del cols

ctPred3.to_csv('ctPred3.csv', header=True, index=False)

## 4-2. NeuralNet modeling3
ctPred3 = pd.read_csv('ctPred3.csv')
# train_test_split
train, test = train_test_split(ctPred3, test_size = 0.2)

# make NN model
NN = keras.Sequential()
NN.add(keras.layers.Input(shape = (train.shape[1]-1)))
NN.add(keras.layers.Dense(18, activation = "relu"))
NN.add(keras.layers.Dense(9, activation = "relu"))
NN.add(keras.layers.Dense(1, activation = None))

Adam = keras.optimizers.Adam(learning_rate = 0.001)

NN.compile(optimizer = Adam, loss = 'mse', metrics = ['mse'])

NN.fit(train.iloc[:,:-1],train.iloc[:,-1], epochs = 800, batch_size=10)

NN.evaluate(test.iloc[:,:-1], test.iloc[:,-1])

(320000**(1/2))/60 # 대략 +-10분정도의 차이를 보임

"""--------------------------------------------------------------------------------------------------------------------
### 5. NN by selected variables (selected user charicteristics + accumulate/recent session variables + recent cooktime)
--------------------------------------------------------------------------------------------------------------------"""
ctPred4 = ctPred3.iloc[:,[1,2,3,4,5,16,20,21,22,23,24,25,26,27,28,29]]

# train_test_split
train, test = train_test_split(ctPred4, test_size = 0.2)

# make NN model
NN = keras.Sequential()
NN.add(keras.layers.Input(shape = (train.shape[1]-1)))
NN.add(keras.layers.Dense(7, activation = "relu"))
NN.add(keras.layers.Dense(1, activation = None))

Adam = keras.optimizers.Adam(learning_rate = 0.001)

NN.compile(optimizer = Adam, loss = 'mse', metrics = ['mse'])

NN.fit(train.iloc[:,:-1],train.iloc[:,-1], epochs = 150, batch_size=10)

NN.evaluate(test.iloc[:,:-1], test.iloc[:,-1])

(600000**(1/2))/60 # 대략 +-12분정도의 차이를 보임

"""------------------------------------------------------------------------------------------------------
### 6. NN by menu (user charicteristics + accumulate/recent session variables + recent cooktime)
------------------------------------------------------------------------------------------------------"""
## 6-1. 요리횟수가 첫번째로 많은 메뉴(군고구마)
ctPred5_7 = ctPred3.loc[ctPred3['Cook_menu'] == 7, ctPred3.columns != 'Cook_menu']

# train_test_split
train, test = train_test_split(ctPred5_7, test_size = 0.2)

# make NN model
NN = keras.Sequential()
NN.add(keras.layers.Input(shape = (train.shape[1]-1)))
NN.add(keras.layers.Dense(18, activation = "relu"))
NN.add(keras.layers.Dense(9, activation = "relu"))
NN.add(keras.layers.Dense(1, activation = None))

Adam = keras.optimizers.Adam(learning_rate = 0.001)

NN.compile(optimizer = Adam, loss = 'mse', metrics = ['mse'])

NN.fit(train.iloc[:,:-1],train.iloc[:,-1], epochs = 300, batch_size=10)

NN.evaluate(test.iloc[:,:-1], test.iloc[:,-1])

## 6-2. 요리횟수가 두번째로 많은 메뉴(냉동밥데우기)
ctPred5_12 = ctPred3.loc[ctPred3['Cook_menu'] == 12, ctPred3.columns != 'Cook_menu']

# train_test_split
train, test = train_test_split(ctPred5_12, test_size = 0.2)

# make NN model
NN = keras.Sequential()
NN.add(keras.layers.Input(shape = (train.shape[1]-1)))
NN.add(keras.layers.Dense(18, activation = "relu"))
NN.add(keras.layers.Dense(9, activation = "relu"))
NN.add(keras.layers.Dense(1, activation = None))

Adam = keras.optimizers.Adam(learning_rate = 0.001)

NN.compile(optimizer = Adam, loss = 'mse', metrics = ['mse'])

NN.fit(train.iloc[:,:-1],train.iloc[:,-1], epochs = 100, batch_size=10)

NN.evaluate(test.iloc[:,:-1], test.iloc[:,-1])

"""------------------------------------------------------------------------------------------------------
### 7. mse of naive rule
------------------------------------------------------------------------------------------------------"""
# 이전에 해당 메뉴를 요리한 경험이 있으면 이전 설정시간을,
# 아니면 해당 메뉴의 expected cooktime을 추천하는 방식

## 7-1. 모든 메뉴에 대한 naive rule 적용
(np.mean((ctPred3['Y'] - ctPred3['recent_ct'])**(2)))**(1/2)/60

# naive한 방식으로 모든 메뉴를 예측하면 22분정도 오차

## 7-2. 개별 메뉴에 대한 naive rule 적용
# 군고구마
df = ctPred5_7.copy(deep=True)
df.loc[df['recent_ct'] == 0,'recent_ct'] = 1725.72
np.mean((df.Y-df.recent_ct)**2)/60

# 냉동밥데우기
df = ctPred5_12.copy(deep=True)
df.loc[df['recent_ct'] == 0,'recent_ct'] = 186.44
np.mean((df.Y-df.recent_ct)**2)/60

# naive한 방식으로 예측하면 23분, 15분정도 오차