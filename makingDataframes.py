import os
import pandas as pd

os.chdir('C:/Users/Kyeongjun/Desktop/oven')

### oven_clustering.csv
oven = pd.read_csv("Oven_sample.csv", encoding = 'euc-kr')
## 총 사용시간 계산 함수 (*1)
def usingtime(df,contvar):
    var = list(set(df[contvar]))
    H, M, S = [], [], []
    for i in var:
        hour = df.loc[df[contvar] == i, 'Cook_hour'].sum()
        minute = df.loc[df[contvar] == i, 'Cook_min'].sum()
        sec = df.loc[df[contvar] == i, 'Cook_sec'].sum()
        
        minute += sec//60; sec = sec%60;
        hour += minute//60; minute = minute%60
        
        H.append(hour); M.append(minute); S.append(sec);

    returndf = pd.DataFrame(list(zip(var,H,M,S)), columns = ['Control_var','Cook_hour', 'Cook_min', 'Cook_sec'])
    return returndf.sort_values(by='Control_var')

# DVICE_ID별로 각 변수를 계산하여 pd.df로 만들고 이를 합치는 방식으로 진행
# 취소, 재시작을 고려할 수 없어서 처음 설정 시간으로 계산하도록 함
oven_pp = oven.iloc[:,[2,3,4,5,7,8,9,11]]
## 변수선택 ('DEVICE_ID', 'EVENT','Cook_State', 'Cook_Mode', 'Cook_hour', 'Cook_min', 'Cook_sec','MONTH')
oven_pp = oven_pp.loc[oven_pp['EVENT'] == '요리시작',:]

## Total_Usingtime (*1)usingtime 함수 사용
total_usingtime = usingtime(oven_pp,'DEVICE_ID')
#total_usingtime.to_csv('total_usingtime.csv', header=True, index=False)

## Self/AutoCook_freq
oven_pp = oven_pp.iloc[:,[0,3,7]]
oven_pp = oven_pp.loc[oven_pp['Cook_Mode'] != '대기',]

oven_pp['auto'] = 0

for i in range(len(oven_pp['DEVICE_ID'])) : # cook_mode에서 '자동' 키워드 유무로 요리 구분
    if ('자동' in oven_pp.iloc[i,1]) :
        oven_pp.iloc[i,3] = 1
    else :
        oven_pp.iloc[i,3] = 0

def usingfreq1(df,contvar, countvar) :
    var = list(set(df[contvar]))
    f0, f1 = [] , []
    for i in var :            
        f_0 = df.loc[df[contvar] == i, countvar].sum()
        f_1 = len(df.loc[df[contvar] == i, countvar]) - f_0
    
        f0.append(f_0); f1.append(f_1);
        
    returndf = pd.DataFrame(list(zip(var,f0,f1)), columns = ['Control_var','AutoCook_freq', 'SelfCook_freq'])
    return returndf

auto_self_freq = usingfreq1(oven_pp, 'DEVICE_ID', 'auto')
#auto_self_freq.to_csv("auto_self_freq.csv", header = True, index = False)

## Micro/Oven/Others_freq
oven_pp['MOO'] = 0
for i in range(len(oven_pp['DEVICE_ID'])) : # cook_mode를 이용하여 레인지/오븐/기타 요리 구분
    if ('레인지' in oven_pp.iloc[i,1]) :
        oven_pp.iloc[i,4] = 2
    elif ('오븐' in oven_pp.iloc[i,1]) :
        oven_pp.iloc[i,4] = 1
    else :
        oven_pp.iloc[i,4] = 0


def usingfreq2(df,contvar, countvar) :
    var = list(set(df[contvar]))
    f0, f1, f2 = [] , [], []
    for i in var :            
        f_2 = df.loc[(df[contvar] == i) & (df[countvar] == 2), countvar].sum()/2
        f_1 = df.loc[(df[contvar] == i) & (df[countvar] == 1), countvar].sum()
        f_0 = len(df.loc[df[contvar] == i, countvar]) - f_2 - f_1
        
        f0.append(f_0); f1.append(f_1); f2.append(f_2)
        
    returndf = pd.DataFrame(list(zip(var,f2,f1,f0)), columns = ['Control_var','Micro_freq', 'Oven_freq', 'Others_freq'])
    return returndf

MOO_freq = usingfreq2(oven_pp, 'DEVICE_ID', 'MOO')
#MOO_freq.to_csv("MOO_freq.csv", header=True, index=False)

## Season_freq
oven_pp['Season'] = 0
for i in range(len(oven_pp['DEVICE_ID'])) : # MONTH를 이용하여 계절 구분
    if (oven_pp.iloc[i,2][5:] in ['03','04','05']) : # 3-5월 : 봄
        oven_pp.iloc[i,5] = 0
    elif (oven_pp.iloc[i,2][5:] in ['06','07','08']) : # 6-8월 : 여름
        oven_pp.iloc[i,5] = 1
    elif (oven_pp.iloc[i,2][5:] in ['09','10','11']) : # 9-11월 : 가을
        oven_pp.iloc[i,5] = 2
    else :                                              # 12-2월 : 겨울
        oven_pp.iloc[i,5] = 3


def usingfreq3(df,contvar, countvar) :
    var = list(set(df[contvar]))
    f0, f1, f2, f3 = [] , [], [], []
    for i in var :
        f_3 = df.loc[(df[contvar] == i) & (df[countvar] == 3), countvar].sum()/3
        f_2 = df.loc[(df[contvar] == i) & (df[countvar] == 2), countvar].sum()/2
        f_1 = df.loc[(df[contvar] == i) & (df[countvar] == 1), countvar].sum()
        f_0 = len(df.loc[df[contvar] == i, countvar]) - f_2 - f_1
        
        f0.append(f_0); f1.append(f_1); f2.append(f_2); f3.append(f_3)
        
    returndf = pd.DataFrame(list(zip(var,f0,f1,f2,f3)), columns = ['Control_var','Spring_freq', 'Summer_freq', 'Fall_freq','Winter_freq'])
    return returndf

season_freq = usingfreq3(oven_pp, 'DEVICE_ID', 'Season')
#season_freq.to_csv("season_freq.csv", header=True, index=False)

## mealtime_freq
oven_pp2 = oven.iloc[:,[1,2,3,4,5]]
oven_pp2 = oven_pp2.loc[oven_pp2['EVENT'] == '요리시작',:]
oven_pp2 = oven_pp2.loc[oven_pp2['Cook_Mode'] != '대기',]
oven_pp2['mealtime'] = 0

for i in range(len(oven_pp2['DEVICE_ID'])) : # LOCAL_TIME을 이용하여 계절 구분
    if (oven_pp2.iloc[i,0][11:13] in ['02','03','04']) : # 1.5 ~ 5.5시 : bfB
        oven_pp2.iloc[i,5] = 0
    elif (oven_pp2.iloc[i,0][11:13] in ['01']) & (int(oven_pp2.iloc[i,0][14:16])>=30)  :
        oven_pp2.iloc[i,5] = 0
    elif (oven_pp2.iloc[i,0][11:13] in ['01']) & (int(oven_pp2.iloc[i,0][14:16])<30)  :
        oven_pp2.iloc[i,5] = 4
    elif (oven_pp2.iloc[i,0][11:13] in ['06','07','08','09']) : # 5.5 ~ 10.5시 : Breakfast
        oven_pp2.iloc[i,5] = 1
    elif (oven_pp2.iloc[i,0][11:13] in ['05']) & (int(oven_pp2.iloc[i,0][14:16])>=30)  :
        oven_pp2.iloc[i,5] = 1
    elif (oven_pp2.iloc[i,0][11:13] in ['05']) & (int(oven_pp2.iloc[i,0][14:16])<30)  :
        oven_pp2.iloc[i,5] = 0
    elif (oven_pp2.iloc[i,0][11:13] in ['11','12','13','14']) : # 10.5 ~ 15.5시 : Lunch
        oven_pp2.iloc[i,5] = 2
    elif (oven_pp2.iloc[i,0][11:13] in ['10']) & (int(oven_pp2.iloc[i,0][14:16])>=30)  :
        oven_pp2.iloc[i,5] = 2
    elif (oven_pp2.iloc[i,0][11:13] in ['10']) & (int(oven_pp2.iloc[i,0][14:16])<30)  :
        oven_pp2.iloc[i,5] = 1
    elif (oven_pp2.iloc[i,0][11:13] in ['16','17','18','19']) : # 15.5 ~ 20.5시 : Dinner
        oven_pp2.iloc[i,5] = 3
    elif (oven_pp2.iloc[i,0][11:13] in ['15']) & (int(oven_pp2.iloc[i,0][14:16])>=30)  :
        oven_pp2.iloc[i,5] = 3
    elif (oven_pp2.iloc[i,0][11:13] in ['15']) & (int(oven_pp2.iloc[i,0][14:16])<30)  :
        oven_pp2.iloc[i,5] = 2
    elif (oven_pp2.iloc[i,0][11:13] in ['20']) & (int(oven_pp2.iloc[i,0][14:16])>=30)  :
        oven_pp2.iloc[i,5] = 4
    elif (oven_pp2.iloc[i,0][11:13] in ['20']) & (int(oven_pp2.iloc[i,0][14:16])<30)  :
        oven_pp2.iloc[i,5] = 3
    else :                                              # 20.5 ~ 1.5시 : afD
        oven_pp2.iloc[i,5] = 4

def usingfreq4(df,contvar, countvar) :
    var = list(set(df[contvar]))
    f0, f1, f2, f3, f4 = [] , [], [], [], []
    for i in var :
        f_4 = df.loc[(df[contvar] == i) & (df[countvar] == 4), countvar].sum()/4
        f_3 = df.loc[(df[contvar] == i) & (df[countvar] == 3), countvar].sum()/3
        f_2 = df.loc[(df[contvar] == i) & (df[countvar] == 2), countvar].sum()/2
        f_1 = df.loc[(df[contvar] == i) & (df[countvar] == 1), countvar].sum()
        f_0 = len(df.loc[df[contvar] == i, countvar]) - f_2 - f_1
        
        f0.append(f_0); f1.append(f_1); f2.append(f_2); f3.append(f_3), f4.append(f_4)
        
    returndf = pd.DataFrame(list(zip(var,f0,f1,f2,f3,f4)), columns = ['Control_var','bfB_freq', 'Breakfast_freq', 'Lunch_freq','Dinner_freq','afD_freq'])
    return returndf

mealtime_freq = usingfreq4(oven_pp2, 'DEVICE_ID', 'mealtime')
#mealtime_freq.to_csv("mealtime_freq.csv", header=True, index=False)

## Preheat_freq
oven_pp3 = pd.read_csv("oven_pp3.csv")

def usingfreq5(df,contvar, countvar) :
    var = list(set(df[contvar]))
    f0 = []
    for i in var :            
        f_0 = df.loc[df[contvar] == i, countvar].sum()
    
        f0.append(f_0)
        
    returndf = pd.DataFrame(list(zip(var,f0)), columns = ['Control_var','freq'])
    return returndf

preheat_freq = usingfreq5(oven_pp3, 'DEVICE_ID', 'preheat')
#preheat_freq.to_csv("preheat_freq.csv", header=True, index=False)

## Clean_freq
clean_freq = usingfreq5(oven_pp3, 'DEVICE_ID', 'clean')
#clean_freq.to_csv("clean_freq.csv", header=True, index=False)

## Cancle_freq
cancle_freq = usingfreq5(oven_pp3, 'DEVICE_ID', 'cancle')
#cancle_freq.to_csv("cancle_freq.csv", header=True, index=False)

## Restart_freq
restart_freq = usingfreq5(oven_pp3, 'DEVICE_ID', 'restart')
#restart_freq.to_csv("restart_freq.csv", header=True, index=False)

## Error_freq
error_freq = usingfreq5(oven_pp3, 'DEVICE_ID', 'error')
#error_freq.to_csv("error_freq.csv", header=True, index=False)

## MenuSpectrum
oven_pp4 = oven.iloc[:,[2,5,6]]

def spectrum1(df,contvar, countvar) :
    var = list(set(df[contvar]))
    c = []
    for i in var :
        cnt = len(list(set(df.loc[df[contvar] == i, countvar])))
        c.append(cnt)
    
    returndf = pd.DataFrame(list(zip(var,c)), columns = ['Control_var', 'MenuSpectrum'])
    return returndf

menuSpectrum = spectrum1(oven_pp4, 'DEVICE_ID', 'Cook_menu')
#menuSpectrum.to_csv("menuSpectrum.csv", header=True, index = False)

## ModeSpectrum

def spectrum2(df,contvar, countvar) :
    var = list(set(df[contvar]))
    c = []
    for i in var :
        cnt = len(list(set(df.loc[df[contvar] == i, countvar])))
        c.append(cnt)
    
    returndf = pd.DataFrame(list(zip(var,c)), columns = ['Control_var', 'ModeSpectrum'])
    return returndf

modeSpectrum = spectrum2(oven_pp4, 'DEVICE_ID', 'Cook_Mode')
#modeSpectrum.to_csv('modeSpectrum.csv', header=True, index=False)

## make oven_clustering df
total_usingtime = total_usingtime.sort_values(by='Control_var')
total_usingtime['total_usingtime'] = round(total_usingtime['Cook_hour'].astype(int) + total_usingtime['Cook_min'].astype(int)/60 , 2)
auto_self_freq = auto_self_freq.sort_values(by='Control_var')
MOO_freq = MOO_freq.sort_values(by='Control_var')
season_freq = season_freq.sort_values(by='Control_var')
mealtime_freq = mealtime_freq.sort_values(by='Control_var')
preheat_freq = preheat_freq.sort_values(by='Control_var')
clean_freq = clean_freq.sort_values(by='Control_var')
cancle_freq = cancle_freq.sort_values(by='Control_var')
restart_freq = restart_freq.sort_values(by='Control_var')
error_freq = error_freq.sort_values(by='Control_var')
menuSpectrum = menuSpectrum.sort_values(by='Control_var')
modeSpectrum = modeSpectrum.sort_values(by='Control_var')
dev_id = list(set(modeSpectrum.Control_var))
dev_id.sort()

v = list(zip(dev_id, list(total_usingtime.iloc[:,-1]), list(auto_self_freq.iloc[:,-2]), list(auto_self_freq.iloc[:,-1]), list(MOO_freq.iloc[:,-3]), list(MOO_freq.iloc[:,-2]), list(MOO_freq.iloc[:,-1]), list(season_freq.iloc[:,-4]), list(season_freq.iloc[:,-3]), list(season_freq.iloc[:,-2]), list(season_freq.iloc[:,-1]), list(mealtime_freq.iloc[:,-5]), list(mealtime_freq.iloc[:,-4]), list(mealtime_freq.iloc[:,-3]), list(mealtime_freq.iloc[:,-2]), list(mealtime_freq.iloc[:,-1]), list(preheat_freq.iloc[:,-1]), list(clean_freq.iloc[:,-1]), list(cancle_freq.iloc[:,-1]), list(restart_freq.iloc[:,-1]), list(menuSpectrum.iloc[:,-1]), list(modeSpectrum.iloc[:,-1])))

oven_clustering = pd.DataFrame(v, columns = ['DEVICE_ID', 'Cook_hour','AutoCook_freq','SelfCook_freq','Micro_freq','Oven_freq','Others_freq','Spring_freq','Summer_freq','Fall_freq','Winter_freq','bfB_freq','Breakfast_freq','Lunch_freq','Dinner_freq','afD_freq','Preheat_freq','Clean_freq','Cancle_freq','Restart_freq','MenuSpectrum','ModeSpectrum'])
oven_clustering.to_csv('oven_clustering.csv')

### oven_cooktimePred.csv
oven = pd.read_csv("Oven_sample.csv", encoding = 'euc-kr')

### preprocessing
oven = oven.iloc[:,[2,3,5,6,7,8,9]]     # 변수 선택
oven = oven.loc[oven['EVENT'] == '요리시작',]   # 요리 시작 시간만 고려.

# 분석할 메뉴 선택 기준
# 100회 이상 요리된 메뉴
# 10명 이상의 user(device)가 요리한 메뉴
# 기준에 만족하는 메뉴 list 만들기
menu_count = oven.Cook_menu.value_counts()
menu_count = pd.DataFrame(menu_count[menu_count >= 100])

menu100 = list(menu_count.index)

mlist, clist = [], []
for i in list(set(oven['Cook_menu'])) :
    mlist.append(i)
    df = oven.loc[oven['Cook_menu'] == i,]
    dev_count = len(set(df.DEVICE_ID))
    clist.append(dev_count)
mc = pd.DataFrame(list(zip(mlist,clist)), columns = ['menu', 'count'])
del mlist, clist, i, dev_count, df, menu_count

mc = mc.loc[mc['count'] >= 10,]

menu = menu100 + list(mc['menu'])
menu = pd.DataFrame([[x,menu.count(x)] for x in set(menu)], columns = ['menu', 'count'])
menu = menu.loc[menu['count'] == 2,]
menu = list(menu['menu'])
menu.remove('0')
menu.remove('해동')

del mc, menu100

oven = oven.loc[oven['Cook_menu'].isin(menu),]

oven.to_csv('oven_cooktimePred.csv', header=True, index=False, encoding = 'euc-kr')

### session.csv
## making DataFrame
os.chdir("C:/Users/Kyeongjun/Desktop/LG가전데이터")
oven = pd.read_csv("Oven_sample.csv", encoding = 'euc-kr')
oven = oven.iloc[:,1:11]

oven['LOCAL_TIME'] = pd.to_datetime(oven['LOCAL_TIME'], format = '%Y-%m-%d %H:%M:%S')

oven = oven.loc[oven['EVENT'].isin(['요리시작', '예열시작', '청소시작', '요리재시작', '에러 발생']),:]
oven = oven.sort_values(by=['DEVICE_ID','LOCAL_TIME'])

# Session2 : 1시간 기준
oven['Session2'] = 0
oven = oven.reset_index(drop=True)

for i in range(oven.shape[0]-1) :
    if oven.iloc[i+1,1] == oven.iloc[i,1] :
        t = oven.iloc[i+1,0] - oven.iloc[i,0]
        td = (t.days*24*60 + t.seconds/60) - oven.iloc[i,6]*60 - oven.iloc[i,7] - oven.iloc[i,8]/60
        if (td > 60) :
            oven['Session2'][i+1] = oven['Session2'][i]+1
        else :
            oven['Session2'][i+1] = oven['Session2'][i]
    else :
        oven['Session2'][i+1] = 0
    print(i)
    
del i, t, td

oven.to_csv('Session.csv', encoding='euc-kr', header=True, index=False)

### dailytotal.csv

oven = pd.read_csv('Oven_Sample.csv', encoding = 'euc-kr')
oven['Cookingtime'] = oven['Cook_hour']*60*60 + oven['Cook_min']*60 + oven['Cook_sec']    # 요리시간(Cookingtime) 계싼
oven = oven.loc[oven['EVENT'] == '요리시작',]                                             # EVENT가 '요리시작'인 데이터만 추출
oven.loc[oven['Cook_Mode'].isin(['레인지', '레인지 자동']),'Cook_Mode'] = '레인지'         # Mode가 레인지/레인지 자동 -> 레인지
oven.loc[oven['Cook_Mode'].isin(['오븐', '오븐 자동']),'Cook_Mode'] = '오븐'               # Mode가 오븐/오븐 자동 -> 오븐
oven.loc[~oven['Cook_Mode'].isin(['레인지', '오븐']),'Cook_Mode'] = '기타'                # 나머지 -> 기타
oven['LOCAL_TIME'] = pd.to_datetime(oven['LOCAL_TIME'].str[:10])                         # 날짜만 저장

# LOCAL_TIME을 day(0~)로 변경
df = pd.DataFrame()
for i in list(set(oven.DEVICE_ID)) : 
    df0 = oven.loc[oven['DEVICE_ID'] == i,]
    df0 = df0.sort_values(by='LOCAL_TIME')
    refertime = df0.iloc[0,1]
    df0['day'] = 0
    for j in range(df0.shape[0]) :
        df0.iloc[j,13] = (df0.iloc[j,1] - refertime).days
    
    df = df.append(df0)

del df0, refertime, i, j

oven = df.copy(deep=True)

# 변수 선택 (DEVICE_ID, Cook_Mode, Cookingtime, day)
dailytotal = oven.iloc[:,[2,5,12,13]]

dailytotal['Micro'], dailytotal['Oven'], dailytotal['Others'] = 0, 0, 0 

df = pd.DataFrame()
for i in list(set(dailytotal.DEVICE_ID)) :
    df0 = dailytotal.loc[dailytotal['DEVICE_ID'] == i,:]
    Micro_t, Oven_t, Others_t = [], [], []
    for j in list(set(df0.day)) : 
        df1 = df0.loc[df0['day'] == j, ]
        micro = sum(df1.loc[df1['Cook_Mode'] == '레인지','Cookingtime']) if ('레인지' in list(df1.Cook_Mode)) else 0
        oven = sum(df1.loc[df1['Cook_Mode'] == '오븐', 'Cookingtime']) if ('오븐' in list(df1.Cook_Mode)) else 0
        others = sum(df1.loc[df1['Cook_Mode'] == '기타','Cookingtime']) if ('기타' in list(df1.Cook_Mode)) else 0
        Micro_t.append(micro) ; Oven_t.append(oven) ; Others_t.append(others);
    df00 = pd.DataFrame(list(zip(list(set(df0.day)), Micro_t, Oven_t, Others_t)), columns= ['day','micro_t','oven_t','others_t'])
    df00['DEVICE_ID'] = i
    
    df = df.append(df00)

del i, j, df0, Micro_t, Oven_t, Others_t, df1, micro, oven, others, df00

df = df[['DEVICE_ID','day','micro_t','oven_t','others_t']]

df.to_csv('dailytotal.csv', header=True, index=False)

### error_day.csv
oven = pd.read_csv('Oven_Sample_e.csv', encoding = 'euc-kr')

oven = oven.iloc[:,[0,1,2,4,5,6,7,8,9,10,11,12,13,14,15]]       # 변수 선택
oven['LOCAL_TIME'] = pd.to_datetime(oven['LOCAL_TIME'].str[:10])    # 날짜로 변경

# 시, 분, 초 -> 시간으로 병합
oven['Cookingtime'] = oven['Cook_hour']*60*60 + oven['Cook_min']*60 + oven['Cook_sec']
del oven['Cook_hour'], oven['Cook_min'], oven['Cook_sec']

# 요리 시작, 요리재시작, 예열시작, 청소시작
oven = oven.loc[oven['EVENT'].isin(['요리시작','요리재시작','예열시작','청소시작']),]
set(oven.EVENT)

df = pd.DataFrame()

for i in list(set(oven.DEVICE_ID)) : 
    df1 = oven.loc[oven['DEVICE_ID'] == i,]
    time, micro, oven_f, others, auto, self, preheat, clean, restart, menu, mode, day = [],[],[],[],[],[],[],[],[],[],[],[]
    for j in list(set(df1.LOCAL_TIME)) : 
        df2 = df1.loc[df1['LOCAL_TIME'] == j,]
        time.append(sum(df2.Cookingtime))
        micro.append(sum(df2.loc[:,'micro_f']))
        oven_f.append(sum(df2.loc[:,'oven_f']))
        others.append(sum(df2.loc[:,'others_f']))
        auto.append(sum(df2.loc[:,'auto_f']))
        self.append(sum(1 - df2.loc[:,'auto_f']))
        preheat.append(sum(df2.loc[:,'preheat_f']))
        clean.append(sum(df2.loc[:,'clean_f']))
        restart.append(sum(df2.loc[:,'restart_f']))
        menu.append(len(set(df2.Cook_menu)))
        mode.append(len(set(df2.Cook_Mode)))
        day.append(j)
        
    df0 = pd.DataFrame(list(zip(day,time,micro, oven_f, others, auto, self,preheat, clean, restart, menu, mode)), columns = ['day', 'time', 'micro', 'oven', 'others', 'auto', 'self', 'preheat', 'clean', 'restart', 'menu', 'mode'])
    df0['DEVICE_ID'] = i
    
    df = df.append(df0)

del i, j, df1, time, micro, oven_f, others, auto, self, preheat, clean, restart, menu, mode, day, df2, df0

df0 = df.copy(deep=True)
df0 = df[['DEVICE_ID', 'day', 'time', 'micro', 'oven', 'others', 'auto', 'self', 'preheat', 'clean', 'restart', 'menu', 'mode']]

df0 = df0.sort_values(by=['DEVICE_ID','day'])

df0.to_csv('error_day', index = False, header=True)
