# 프로젝트 설명
### 개요
모 전자회사 생활가전 중 레인지오븐의 사용자 데이터를 이용하여 datamining과 여러가지 간단한 데이터 분석과제를 진행
### 데이터
모 전자회사의 oven 사용기록.
데이터 기록시간과 device_id, event, menu, mode, setting time, setting temperature 등
### 분석 내용
- project0 : 각 기록별 요리 설정시간과 요리 종료시의 기록시간 사이에 차이가 발생함을 발견하고 이를 시각화 해봄
- project1 : 데이터에서 계산할 수 있는 여러 변수들을 생성하여 해당 변수들로 device_id를 클러스터링
- project2 : NeuralNetwork 모형을 이용하여 개인별 요리 시간을 예측하는 모델링
- project3 : GESD를 활용한 메뉴별 요리 시간 anomaly detection
- project4 : day별 요리한 메뉴들을 바탕으로 frequent-item mining
- project5 : GESD를 활용하여 cooktime(daily total/레인지/오븐/기타모드 사용시간) anomaly detection과 
             PCA를 이용해서 2 variables의 상태를 고려한 GESD anomaly detection
- project8 : 데이터를 1시간 단위로 session을 구분한 뒤, 에러 발생 하루, 한 세션 이전 상태에 관한 시각화

# 파일 설명

### makingDataframes
 : 1,2,5번 분석과제는 가공된 데이터를 csv파일로 저장한 뒤 읽어서 사용했습니다.
   각각의 분석과제에서 사용된 csv파일을 생성하는 코드입니다.

### project0_TimeDifferenceVisualization
 : 요리 설정 시간과 데이터 기록 시점에 차이가 있는 부분에 대해 시각화 한 내용입니다.

### project1_DeviceClustering
 : 1번과제(device id 클러스터링)에 관한 내용입니다.

### project2_PersonalCooktimePrediction
 : 2번과제(NeuralNetwork 모형을 이용하여 개인별 요리시간을 예측)에 관한 내용입니다.

### project3_MenuCooktimeAnomalydetection
 : 3번과제(GESD를 이용하여 메뉴별 요리시간 anomaly detection)에 관한 내용입니다.

### project4_FrequentItemMining
 : 4번과제(frequent-item mining)에 관한 내용입니다.

### project5_DailyTotalcooktimeAnomalydetection
 : 5번과제(daily cooktime(total, 레인지, 오븐, 기타모드 사용시간) GESD와
   PCA component GESD를 이용한 anomaly detection)에 관한 내용입니다.

### project8_ErrorAnalysis
 : 8번과제(에러 발생 day-1, session-1 시각화)에 관한 내용입니다.
