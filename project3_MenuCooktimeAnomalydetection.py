import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import scipy.stats as stats
%matplotlib inline

from tqdm.notebook import tqdm

os.chdir("C:/Users/USER/Desktop/oven")

"""------------------------------------------------------------------------------------------------------
### 1. cooktime of menus GESD
------------------------------------------------------------------------------------------------------"""
# time 변수 추가한 데이터 사용
df=pd.read_csv('Oven_sample.csv',encoding='cp949',parse_dates=['CREATE_DT','LOCAL_TIME','MONTH'],dayfirst=True)
df['time'] = df['Cook_hour'].astype(int)*60*60 + df['Cook_min'].astype(int)*60 + df['Cook_sec'].astype(int)

# 변수 선택
df=df[['CREATE_DT','DEVICE_ID','EVENT','Cook_State','Cook_Mode','Cook_menu','Target_temp','time']]


## 1-1. GESD function
def grubbs_stat(y):
    std_dev = np.std(y)
    avg_y = np.mean(y)
    abs_val_minus_avg = abs(y - avg_y)
    max_of_deviations = max(abs_val_minus_avg)
    max_ind = np.argmax(abs_val_minus_avg)
    Gcal = max_of_deviations/ std_dev
    print("Grubbs Statistics Value : {}".format(Gcal))
    return Gcal, max_ind

def calculate_critical_value(size, alpha):
    t_dist = stats.t.ppf(1 - alpha / (2 * size), size - 2)
    numerator = (size - 1) * np.sqrt(np.square(t_dist))
    denominator = np.sqrt(size) * np.sqrt(size - 2 + np.square(t_dist))
    critical_value = numerator / denominator
    print("Grubbs Critical Value: {}".format(critical_value))
    return critical_value

def check_G_values(Gs, Gc, inp, max_index):
    if Gs > Gc:
        print('{} is an outlier. G > G-critical: {:.4f} > {:.4f} \n'.format(inp[max_index], Gs, Gc))
    else:
        print('{} is not an outlier. G > G-critical: {:.4f} > {:.4f} \n'.format(inp[max_index], Gs, Gc))
        
def ESD_Test(input_series, alpha, max_outliers):
    for iterations in range(max_outliers):
        Gcritical = calculate_critical_value(len(input_series), alpha)
        Gstat, max_index = grubbs_stat(input_series)
        check_G_values(Gstat, Gcritical, input_series, max_index)
        input_series = np.delete(input_series, max_index)
    
np.unique(df['Cook_menu']) # 메뉴 확인

## 1-2. preprocessing
# 요리 메뉴 전처리
df=df[~df['Cook_menu'].isin(['0','My_Recipe','스팀청소','스팀발생기세정','요리취소','잔수제거','조리실건조','탈취','해동'])].reset_index(drop=True)

df['Cook_menu'].value_counts()[:70] # 사용된 요리 메뉴

# 요리재시작, 요리시작인 경우만 선택
df=df[df['EVENT'].isin(['요리재시작','요리시작'])].reset_index(drop=True)
df

## 1-3. 사용 수가 중위권인 메뉴 GESD & histogram
# 장어양념구이
df1=df[df['Cook_menu']=="장어양념구이"].reset_index(drop=True)
y=df1['time']
ESD_Test(np.array(y),0.05,5)

plt.hist(y)

# 닭가슴살
df2=df[df['Cook_menu']=="닭가슴살"].reset_index(drop=True)
y=df2['time']
ESD_Test(np.array(y),0.05,5)

plt.hist(y)

# 육포
df3=df[df['Cook_menu']=="육포"].reset_index(drop=True)
y=df3['time']
ESD_Test(np.array(y),0.05,5)

plt.hist(y)

## 1-4. 사용 수가 상위권인 메뉴 GESD & histogram
# 군고구마
df1=df[df['Cook_menu']=='군고구마'].reset_index(drop=True)
y=df1['time']
ESD_Test(np.array(y),0.05,5)

plt.hist(y)

# 식빵
df2=df[df['Cook_menu']=='식빵'].reset_index(drop=True)
y2=df2['time']
ESD_Test(np.array(y2),0.05,5)

plt.hist(y2)

# 냉동밥데우기
df3=df[df['Cook_menu']=='냉동밥데우기'].reset_index(drop=True)
y3=df3['time']
ESD_Test(np.array(y3),0.05,5)

plt.hist(y3)

# 냉장밥데우기
df4=df[df['Cook_menu']=='냉장밥데우기'].reset_index(drop=True)
y4=df4['time']
ESD_Test(np.array(y4),0.05,5)

plt.hist(y4)

# 메뉴별로 파악해봤을때 요리 시간에서 이상치가 나타나는 경우는 찾지 못함
