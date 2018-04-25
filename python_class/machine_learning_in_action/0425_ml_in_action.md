## 머신러닝 인 액션
### 1장. 기계 학습 기초
----------------------------------------------------------------------
>###    1. 머신러닝의 정의 및 중요성
        - https://en.wikipedia.org/wiki/Machine_learning
        - http://www.sas.com/en_us/insights/analytics/machine-learning.html
        - http://whatis.techtarget.com/definition/machine-learning
        - https://www.coursera.org/learn/machine-learning
        - http://www.whydsp.org/237 : 기계학습 / 머신러닝 기초 ( Machine Learning Basics )
>###    2. 머신러닝 기법
        1. 지도학습
            - 정답이 있는 학습 방법
            - 즉, correct outputs를 예상하는 학습 방법
            - 알고리즘에 무엇을 예측할 것인지 제공한다.
            - 분류, 회귀 방식
        2. unsupervised learning(비지도 학습)
            - right answer가 없는 학습
            - 주어진 데이터에 분류 항목 표시나 목적 변수가 없다.
            - 밀도 추정, 군집화 방식.
        3. 올바른 알고리즘 선정
            1) 목적 고려, 얻고자 하는 것을 위해 무엇을 시도할 것인가?
                - 목적 값 예측, 예견을 원하는 경우 : 지도학습 방법
                - 아니라면 비지도학습, 데이터가 이산적 무리에 알맞은지 알아보고자 하는 경우 : 군집화
                - 비지도학습 중, 각각의 무리에 알맞는 정도를 수치로 평가하는 경우 : 밀도 추정 알고리즘 
            2) 보유한 데이터 고려
                - 속성이 명목형인가, 연속형인가
                - 속성 내에 누락값 여부
                - 누락값이 있는 경우 누락 상황 인지
                - 오류 데이터 존재 여부
                - 이상치의 여부
            3) 알고리즘 cheet sheet
                - https://docs.microsoft.com/ko-kr/azure/machine-learning/studio/algorithm-choice
                - http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
        4. key terminology
            - feature : 특성. 데이터가 가지고 있는 속성들 중 특별한 부분
            - 수치형 데이터, 명목형 데이터(연속적, 이산적 데이터)
            - Classification: 분류
            - Class: 분류 항목
            - Training set
            - Training examples
            - Target variable (통계: dependent variable)
            - Test set
            - Knowledge representation: 지식 표현