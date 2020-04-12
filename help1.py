'''
최대 팀 인원: 5명

최대 제출 횟수: 165회

일일 최대 제출: 3회

1. 평가

평가 지표는 MAE를 F1 score로 나눈 값입니다. 이 때 MAE는 실제 값이 0.1 이상인 픽셀에 대해서만 산출하며, F1 score는 해당 픽셀의 강수량이 0.1 이상이면 1, 0.1 미만이면 0으로 변환 후 1에 대해 F1 score를 산출합니다. 실제 값에 결측치(-9999.xxx)가 포함된 경우, 결측치가 포함된 픽셀은 F1 score 계산에서 제외 합니다. 
'''


import numpy as np
from sklearn.metrics import f1_score


def mae_over_fscore(y_true, y_pred):
    '''
    y_true: sample_submission.csv 형태의 실제 값
    y_pred: sample_submission.csv 형태의 예측 값
    '''


    y_true = np.array(y_true)
    y_true = y_true.reshape(1, -1)[0]  
    
    y_pred = np.array(y_pred)
    y_pred = y_pred.reshape(1, -1)[0]
    
    # 실제값이 0.1 이상인 픽셀의 위치 확인
    IsGreaterThanEqualTo_PointOne = y_true >= 0.1
    
    # 실제 값에 결측값이 없는 픽셀의 위치 확인 
    IsNotMissing = y_true >= 0
    
    # mae 계산
    mae = np.mean(np.abs(y_true[IsGreaterThanEqualTo_PointOne] - y_pred[IsGreaterThanEqualTo_PointOne]))
    
    # f1_score 계산 위해, 실제값에 결측값이 없는 픽셀에 대해 1과 0으로 값 변환
    y_true = np.where(y_true[IsNotMissing] >= 0.1, 1, 0)
    
    y_pred = np.where(y_pred[IsNotMissing] >= 0.1, 1, 0)
    
    # f1_score 계산    
    f_score = f1_score(y_true, y_pred) 
    
    # f1_score가 0일 나올 경우를 대비하여 소량의 값 (1e-07) 추가 
    return mae / (f_score + 1e-07) 

'''
A. 가채점 순위 (Public Score) : test 데이터의 30% 데이터로 채점합니다.

B. 최종 순위 (Private Score) : Public Score에서 사용하지 않은 나머지 70% 데이터로 채점합니다. 리더보드 운영 기간 중에는 확인할 수 없으며, 대회 종료 후에 공개됩니다.

C. 최종 순위는 선택된 파일 중에서 채점되므로, 참가자는 제출 창에서 자신이 최종적으로 채점 받고 싶은 파일을 선택해야 합니다.

D. 2020년 05월 25일 17:59 리더보드 운영 종료 이후 Private Score 랭킹이 가장 높은 참가자 6팀은 2020년 06월 01일 23:59 까지 양식에 맞는 코드와 함께 코드 내용을 설명하는 PPT를 제출합니다. (대회 종료 후 
dacon@dacon.io
를 통해 안내드릴 예정입니다.)

E. 대회 직후 공개되는 Private Score 랭킹은 최종 순위가 아니며 코드 검증 후 최종 수상자가 결정 됩니다.

F. 최종 수상자는 오프라인 시상식 참여 또는 솔루션 및 코드 설명 영상 제출 중 최소 하나를 택하셔야 합니다.



2. 외부 데이터

본 대회에서는 외부 데이터 및 pretrained 모델 사용이 가능합니다.

A. 외부 데이터 및 pretrained 모델은 사용에 법적인 제약이 없어야 합니다.

B. 다운로드를 받은 경우 외부데이터 및 pretrained 모델에 대한 링크를 명시해야 합니다.

C. 크롤링을 실시한 경우 크롤링 코드를 제출해야 합니다.

D. data leakage 문제가 없어야 합니다. 모델 훈련에는 2019년도 이전의 데이터만 활용 가능하며, pretrained 모델 또한 2019년 이전에 생성된 데이터로 훈련 된 것이어야 합니다.



3. 참가 방법

- 개인 참가 방법 : 팀 신청 없이, 자유롭게 제출 창에서 제출 가능  

- 팀 참가 방법 : 팀 배너에서 가능, 상세 내용은 팀 배너에서 팀 병합 정책을 확인 부탁드립니다.

* 하나의 대회에는 하나의 팀으로만 등록이 가능합니다.

* 팀의 수상 요건 충족 시 팀의 대표가 수상하게 됩니다.



4. 코드

1) 입상자는 코드 제출 필수. 제출 코드는 예측 결과를 리더보드 점수로 복원할 수 있어야 함

2) 코드 제출시 확장자가 R user는 R or .rmd. Python user는 .py or .ipynb

3) 코드에 ‘/data’ 데이터 입/출력 경로 포함 제출

4) 전체 프로세스를 일목요연하게 정리하여 주석을 포함하여 하나의 파일로 제출

5) 모든 코드는 오류 없이 실행되어야 함(라이브러리 로딩 코드 포함되어야 함).

6) 코드와 주석의 인코딩은 모두 UTF-8을 사용하여야 함

7) 코드 제출 시 데이콘에서 안내한 양식에 맞추어 제출하여야 함



5. 토론(질문)

해당 대회에서는 대회 운영 및 데이터 이상에 관련된 질문 외에는 답변 드리지 않을 예정입니다. 결측치 처리 방법, 모델 구성 방법 등등 대회 운영 및 데이터 이상 외 질문은 토론 페이지를 통해 자유롭게 토론해 주시기 바랍니다.

* 데이콘 답변을 요청하는 경우 토론 제목에 [DACON 답변 요청] 문구를 넣어 질문을 올려 주시기바랍니다. 예) [DACON 답변 요청] 시상식은 언제 열리나요?
    '''