'''
[Dacon] AI프렌즈 시즌2 위성관측 데이터 활용 강수량 산출 대회
_ (팀명)
2020년 04월 10일 (제출날짜)
모델링 코드 작성방법
A 코드 관련

1) 입상자는 코드 제출 필수. 제출 코드는 예측 결과를 리더보드 점수로 복원할 수 있어야 함

2) 코드 제출시 확장자가 R user는 R or .rmd. Python user는 .py or .ipynb

3) 코드에 ‘/data’ 데이터 입/출력 경로 포함 제출 or R의 경우 setwd(" "), python의 경우 os.chdir을 활용하여 경로 통일

4) 전체 프로세스를 일목요연하게 정리하여 주석을 포함하여 하나의 파일로 제출

5) 모든 코드는 오류 없이 실행되어야 함(라이브러리 로딩 코드 포함되어야 함).

6) 코드와 주석의 인코딩은 모두 UTF-8을 사용하여야 함

B 외부 데이터

1) 본 대회에서는 외부 데이터 및 pretrained 모델 사용이 가능합니다.

2) 외부 데이터 및 pretrained 모델은 사용에 법적인 제약이 없어야 합니다.

3) 다운로드를 받은 경우 외부데이터 및 pretrained 모델에 대한 링크를 명시해야 합니다.

4) 크롤링을 실시한 경우 크롤링 코드를 제출해야 합니다.

5) data leakage 문제가 없어야 합니다. 모델 훈련에는 2019년도 이전의 데이터만 활용 가능하며, pretrained 모델 또한 2019년 이전에 생성된 데이터로 훈련 된 것이어야 합니다.


1. 테스트데이터를 다운 
https://dacon.io/m/competitions/official/235591/data/
train.zip (5.84 GB)
test.zip (185 MB)
sample_submission.csv (14.7 MB)
로컬에 압축을 푼다.
home\train\subset_XXXXXX_XX.npy    

2.환경설정 
..각자 PC의 환경마다 다름...
..INTEL i7 64bit

tensorflow 1.4
numpy 1.4
python3.6.2
...

...

#1. 라이브러리 및 데이터
#Library & Data
commit test

'''
import glob
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import random
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Conv2DTranspose, MaxPooling2D, BatchNormalization, Activation, concatenate, Input, GlobalAveragePooling2D
from tensorflow.keras import Model
import warnings
 
warnings.filterwarnings("ignore")
# 재생산성을 위해 시드 고정
np.random.seed(7)
random.seed(7)
tf.random.set_random_seed(7)
'''
밝기 온도 채널만 사용해보기 위해 0~8 채널만 불러오기
약 7만장의 전체 데이터를 사용하지 않고, 50개 이상의 픽셀에 강수량이 기록되어 있는 이미지만 사용해보기
'''
def trainGenerator():
    
    train_path = 'train'
    train_files = sorted(glob.glob(train_path + '/*'))
    
    for file in train_files:
        
        dataset = np.load(file)
        
        target= dataset[:,:,-1].reshape(40,40,1)
        cutoff_labels = np.where(target < 0, 0, target)
        feature = dataset[:,:,:9]
        
        if (cutoff_labels > 0).sum() < 50:
            
            continue

        yield (feature, cutoff_labels)
        
train_dataset = tf.data.Dataset.from_generator(trainGenerator, (tf.float32, tf.float32), (tf.TensorShape([40,40,9]),tf.TensorShape([40,40,1])))
train_dataset = train_dataset.batch(512).prefetch(1)
test_path = 'test'
test_files = sorted(glob.glob(test_path + '/*'))

X_test = []

for file in tqdm(test_files, desc = 'test'):
    
    data = np.load(file)
    
    X_test.append(data[:,:,:9])
                  
X_test = np.array(X_test)

#test: 100%|██████████| 2416/2416 [00:01<00:00, 1830.06it/s]

'''
2. 데이터 전처리
Data Cleansing & Pre-Processing
데이터 불러올 시 전처리 로직 적용
강수량에 결측값은 0으로 일괄 대체

3. 탐색적 자료분석
Exploratory Data Analysis
'''
import seaborn as sns
color_map = plt.cm.get_cmap('RdBu')
color_map = color_map.reversed()
image_sample = np.load('train/subset_010462_02.npy')
#밝기 온도와 강수량과의 관계 확인해보기
plt.style.use('fivethirtyeight')
plt.figure(figsize=(30, 30))

for i in range(9):
    plt.subplot(1,10,i+1)
    plt.imshow(image_sample[:, :, i], cmap=color_map)

plt.subplot(1,10,10)
plt.imshow(image_sample[:,:,-1], cmap = color_map)
plt.show()

'''
4. 변수 선택 및 모델 구축
Feature Engineering & Initial Modeling
모델 구축
활용한 코드1
클릭
https://www.kaggle.com/kmader/baseline-u-net-model-part-1

<- ctrl 누른 상태에서 왼쪽 마우스 클릭
활용한 코드2
클릭
<- ctrl 누른 상태에서 왼쪽 마우스 클릭
https://www.kaggle.com/phoenigs/u-net-dropout-augmentation-stratification
'''

def build_model(input_layer, start_neurons):
    
    # 40 x 40 -> 20 x 20
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_layer)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2))(pool1)
    pool1 = Dropout(0.25)(pool1)

    # 20 x 20 -> 10 x 10
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D((2, 2))(pool2)
    pool2 = Dropout(0.25)(pool2)

    # 10 x 10 
    convm = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)

    # 10 x 10 -> 20 x 20
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.25)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = BatchNormalization()(uconv2)

    # 20 x 20 -> 40 x 40
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.25)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = BatchNormalization()(uconv1)
    uconv1 = Dropout(0.25)(uconv1)
    output_layer = Conv2D(1, (1,1), padding="same", activation='relu')(uconv1)
    
    return output_layer

input_layer = Input((40, 40, 9))
output_layer = build_model(input_layer, 32)
model = Model(input_layer, output_layer)
'''
Loss fuction 정의
활용한 코드
클릭
<- ctrl 누른 상태에서 왼쪽 마우스 클릭
TF에 custom metrics 적용하기
euphoria
#!pip install tensorflow == 2.1.0
#!pip install tensorflow-gpu
import numpy as np
import tensorflow as tf
평가 지표는 실제값과 1 미만의 차이에 대해서는 패널티를 주지 않는 MSE입니다. 센서의 측정오차로 인해, 1 미만의 차이로 예측을 한 값에 대해서는 패널티를 부여하지 않습니다. (대회 설명 참고)
데이콘에서 정의한 metric 함수를 이용하여 tf 혹은 tf.keras에서 사용할 수 있는 코드로 커스텀하였습니다.
이때, tensorflow 1.x과 2.x에서 사용하는 함수가 달라 각자 버전에 맞게 사용하시면 됩니다.
def mse_AIFrenz(y_true, y_pred):
    diff = abs(y_true - y_pred)
    less_then_one = np.where(diff < 1, 0, diff)
    # multi-column일 경우에도 계산 할 수 있도록 np.average를 한번 더 씌움
    try:
        score = np.average(np.average(less_then_one ** 2, axis = 0))
    except ValueError:
        score = mean_squared_error(y_true, y_pred)
    return score
def mse_keras(y_true, y_pred):
    score = tf.py_function(func=mse_AIFrenz, inp=[y_true, y_pred], Tout=tf.float32,  name='custom_mse') # tf 2.x
    #score = tf.py_func( lambda y_true, y_pred : mse_AIFrenz(y_true, y_pred) , [y_true, y_pred], 'float32', stateful = False, name = 'custom_mse' ) # tf 1.x
    return score
구축한 모델 컴파일 시 metrics를 추가하면 됩니다!
model.compile(optimizer='adam', loss='mse', metrics=[mse_keras])

'''
from sklearn.metrics import f1_score

def mae(y_true, y_pred) :
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    y_true = y_true.reshape(1, -1)[0]
    
    y_pred = y_pred.reshape(1, -1)[0]
    
    over_threshold = y_true >= 0.1
    
    return np.mean(np.abs(y_true[over_threshold] - y_pred[over_threshold]))

def fscore(y_true, y_pred):
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    y_true = y_true.reshape(1, -1)[0]
    
    y_pred = y_pred.reshape(1, -1)[0]
    
    remove_NAs = y_true >= 0
    
    y_true = np.where(y_true[remove_NAs] >= 0.1, 1, 0)
    
    y_pred = np.where(y_pred[remove_NAs] >= 0.1, 1, 0)
    
    return(f1_score(y_true, y_pred))

def maeOverFscore(y_true, y_pred):
    
    return mae(y_true, y_pred) / (fscore(y_true, y_pred) + 1e-07)

def fscore_keras(y_true, y_pred):
    score = tf.py_function(func=fscore, inp=[y_true, y_pred], Tout=tf.float32, name='fscore_keras')
    return score

def maeOverFscore_keras(y_true, y_pred):
    score = tf.py_function(func=maeOverFscore, inp=[y_true, y_pred], Tout=tf.float32,  name='custom_mse') 
    return score
model.compile(loss="mae", optimizer="adam", metrics=[maeOverFscore_keras, fscore_keras])
'''
5. 모델 학습 및 검증
Model Tuning & Evaluation
'''
print("========train_dataset=================")
##print(train_dataset)
print("====================================")

##model_history = model.fit(train_dataset, epochs = 2, verbose=1)

'''
Epoch 1/2
61/61 [==============================] - 60s 988ms/step - loss: 0.4086 - maeOverFscore_keras: 5.3017 - fscore_keras: 0.4047
Epoch 2/2
61/61 [==============================] - 44s 714ms/step - loss: 0.2916 - maeOverFscore_keras: 3.3510 - fscore_keras: 0.5401
'''
pred = model.predict(X_test)
submission = pd.read_csv('sample_submission.csv')
submission.iloc[:,1:] = pred.reshape(-1, 1600)
submission.to_csv('Dacon_baseline.csv', index = False)

'''
6. 결과 및 결언
Conclusion & Discussion
'''