import numpy as np
import pandas as pd
import os
import random
from itertools import product 
from itertools import permutations 
from itertools import combinations
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl  # 기본 설정 만지는 용도
import matplotlib.pyplot as plt  # 그래프 그리는 용도
import matplotlib.font_manager as fm 
import scipy.stats as stats
%matplotlib inline

from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx
from google.colab import drive

path1='/content/drive/MyDrive/oven/'
drive.mount('/content/drive')

#나눔폰트 설치되있는지 확인
sys_font=fm.findSystemFonts()
print(f"sys_font number: {len(sys_font)}")
print(sys_font)

nanum_font = [f for f in sys_font if 'Nanum' in f]
print(f"nanum_font number: {len(nanum_font)}")

#나눔폰트 설치
!apt-get update -qq
!apt-get install fonts-nanum* -qq

#나눔폰트가 잘 설치됐는지 확인
sys_font=fm.findSystemFonts()
print(f"sys_font number: {len(sys_font)}")

nanum_font = [f for f in sys_font if 'Nanum' in f]
print(f"nanum_font number: {len(nanum_font)}")

#나눔폰트 경로 확인
nanum_font

#파이썬 버전과 현재 설정되어있는 폰트 확인
!python --version
def current_font():
  print(f"설정 폰트 글꼴: {plt.rcParams['font.family']}, 설정 폰트 사이즈: {plt.rcParams['font.size']}") 
        
current_font()

#폰트 설정
path = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'  
font_name = fm.FontProperties(fname=path, size=10).get_name()
print(font_name)
plt.rc('font', family=font_name)

fm._rebuild()

data=pd.read_csv(path1+'Oven_sample.csv',encoding='cp949')

"""------------------------------------------------------------------------------------------------------
### 1. visualizing frequent-items (menus)
------------------------------------------------------------------------------------------------------"""
## 1-1. preprocessing
#요리 메뉴 전처리
data=data[~data['Cook_menu'].isin(['0','My_Recipe','스팀발생기세정','스팀청소','요리취소','잔수제거','조리실건조','탈취','해동'])].reset_index(drop=True)

#날 별로 분석하기 위해 day변수 생성
data['day']=pd.Series()

for i in tqdm(range(len(data))):
    data.loc[i,'day'] = str(data.loc[i,'LOCAL_TIME'])[:10]

data=data[['day','DEVICE_ID','Cook_menu']] # 변수선택

#column은 unique한 menu, row는 day인 데이터프레임 생성
df=pd.DataFrame(index=np.unique(data['day']),columns=np.unique(data['Cook_menu']))

#apriori 알고리즘에 들어가는 데이터프레임 생성
for day in tqdm(df.index):
  df_day=data[data['day']==day].reset_index(drop=True)
  unique_menu=np.unique(df_day['Cook_menu'])

  for menu in df.columns:        
    if menu in unique_menu:
      df.loc[day,menu]=1
    else:
      df.loc[day,menu]=0
      
## 1-2. 최소지지도 0.01로 설정
frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)
frequent_itemsets['length']=frequent_itemsets.itemsets.apply(lambda x: len(x))

frequent_itemsets

## 1-3. 아이템 조합이 2,3개인 데이터 시각화
# 2,3개 메뉴 조합 데이터 추출
df2=frequent_itemsets[(frequent_itemsets['length']==2) | (frequent_itemsets['length']==3)].reset_index(drop=True)

print("아이템 길이가 2이거나 3인 data 수: ",len(df2))
print("전체 데이터 수: ",len(frequent_itemsets))

## 1-4. visualization
G2=nx.Graph()
ar2=df2['itemsets']
G2.add_edges_from(ar2[:30])

pr2=nx.pagerank(G2)
nsize2=np.array([v for v in pr2.values()])
nsize2=2000*(nsize2-min(nsize2))/(max(nsize2)-min(nsize2))

pos2=nx.planar_layout(G2)

plt.figure(figsize=(16,12))
plt.axis('off')
nx.draw_networkx(G2,font_size=16,font_family='NanumBarunGothic',pos=pos2,node_color=list(pr2.values()),node_size=nsize2,
                 alpha=0.7,edge_color='.5',cmap=plt.cm.YlGn)

"""------------------------------------------------------------------------------------------------------
### 2. association rules
------------------------------------------------------------------------------------------------------"""

## 2-1. frequent-item sorted by support 
frequent_itemsets.sort_values(by='support',ascending=False).head(20)

## 2-2. association rules (confidence >=0.3, sorted by confidence, support, lift)
df_con=association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3).sort_values(by=['confidence','support','lift'],ascending=False)
df_con.head(30)

## 2-3. association rules sorted by lift
association_rules(frequent_itemsets, metric="lift", min_threshold=1)
