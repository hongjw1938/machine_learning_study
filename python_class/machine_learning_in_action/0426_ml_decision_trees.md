##  Splitting datasets one feature at a time: decision trees(의사 결정 트리)
------------------------------------------------------------
>###    의사결정 트리 (Decision Trees)
        * 의사결정 트리는 마치 스무고개 게임처럼 동작한다.
        * Decision Tree Flowchart (그림 3.1 p.49)
            - Decision Block (의사결정 블록, 사각형)
            - Terminal Block (단말 블록, 타원형)
            - Branch (가지)
        * 장점
            - 적은 계산 비용
            - 이해하기 쉬운 학습 결과
            - 누락된 값 있어도 처리 가능
            - 분류와 무관한 특징도 처리 가능
        * 단점
            - 과적합(overfitting)되기 쉬움: 너무 복잡한 의사결정 트리
        * 적용
            - 수치형 값, 명목형 값
>###    3.1 Tree construction
        * ID3 알고리즘
            1. 데이터를 가장 잘 나눌 수 있는 특징을 먼저 찾아서 데이터 집합을 하위 집합으로 분할
                - 정보 이득(Information Gain)이 가장 큰 특징
                - 엔트로피(Entopy)가 가장 크게 낮아지는 특징
            2. 해당 특징을 포함하는 노드 생성
            3. 하위 집합의 모든 데이터가 같은 클래스에 속하면 해당 하위 집합에 대한 분류 종료
            4. 2의 경우가 아니라면 이 하위 집합에 대해 1을 적용
            5. 모든 데이터가 분류될 때까지(= 모든 하위 집합에 대해) 1~4 반복
                - 재귀적 방법으로 해결
                * https://en.wikipedia.org/wiki/ID3_algorithm
        * General approach to Dicision Tree
        1. Collect:
            - 모든 방법
        2. Prepare:
            - 명목형 값
            - 연속형 값(수치형)은 양자화를 통해 이산형 값으로 변환
        3. Analyze:
            - 모든 방법
            - 트리를 구성한 후 시각적으로 검토
        4. Train:
            - 트리 데이터 구조를 구성
        5. Test: 
            - 학습된 트리로 오류율(error rate) 계산
        6. Use:
            - 모든 지도학습에 사용 가능
            - 대개 데이터를 더 잘 이해하기 위해 사용
        * 양자화(Quantization)
            - https://ko.wikipedia.org/wiki/%EC%96%91%EC%9E%90%ED%99%94_(%EC%A0%95%EB%B3%B4_%EC%9D%B4%EB%A1%A0)
            - http://www.ktword.co.kr/abbr_view.php?m_temp1=911
        * 의사결정 트리 알고리즘
            - https://ko.wikipedia.org/wiki/%EA%B2%B0%EC%A0%95_%ED%8A%B8%EB%A6%AC_%ED%95%99%EC%8A%B5%EB%B2%95
            * ID3 (Iterative Dichotomiser 3)
            * C4.5 (successor of ID3)
            * C5.0 (successor of ID4)
            * CART (Classification And Regression Tree)
            * CHAID (CHi-squared Automatic Interaction Detector)
            * MARS (Multivariate adaptive regression splines)
            * 조건부 추론 트리 (Conditional Inference Trees)
        * 가정 적합한 분할 기준을 선택하는 방법
            - 정보 이득
            - 지니 불순도(Gini Impurity)
            - 분산 감소
        3.1.1 Information gain
            - 데이터를 분할하기 이전과 이후의 정보량(엔트로피) 변화
            - 정보 이득이 가장 큰 특징에 대해 분할 수행
            - 정보 이득으로 정보의 불확실성(엔트로피) 감소
        * 개별 정보량과 엔트로피 (p.53~54)
            * 개별 정보량
                - 확률이 낮을수록 개별 정보량은 커진다 == 엔트로피가 커지는데 기여
                - 로그의 결과에 -1을 곱한 이유
            * 밑이 2
                - 정보를 전달(표현)하는데 몇 자리 2진수(몇 비트)면 충분한가
            * 엔트로피
                - 정보에 대한 기댓값
                - 불확실한 정도, 무질서 정도
                - 확률이 낮은 사건이 많을수록 정보의 엔트로피(불확실성)이 커진다
                - 정보의 불확실성(엔트로피)가 높다
                - 어떤 값(정보)가 나올 지 알기 힘들다
            * 엔트로피가 높은 원인
                - 모든 사건의 확률이 균등하다
                - 확률이 낮은 사건이 많다
                - 정보가 다양하다
            * http://leosworld.tistory.com/8
            * https://ko.wikipedia.org/wiki/%EC%A0%95%EB%B3%B4_%EC%97%94%ED%8A%B8%EB%A1%9C%ED%94%BC
        3.1.2 Splitting the dataset
            - dataSet: 분할하고자 하는 데이터 집합
            - axis: 특징의 인덱스
            - value: 특징의 값
        * 분할하기전 엔트로피
            - 0.9709505944546686
        * 0번 특징으로 분할
          * 0번 특징이 0인 그룹 ==> 'no' 2개 ==> 정보가 없다
            - 엔트로피: 0
          * 0번 특징이 1인 그룹 ==> 'yes' 2개, 'no' 1개 ==> 확률이 2/3, 1/3
            - 엔트로피: 0.9182958340544896
          * 0번 특징으로 분할된 두 그룹에 대한 엔트로피의 기댓값
            - 2/5 \* 0.0 + 3/5 * 0.9182958340544896 = 0.5509775004326937
        * 1번 특징으로 분할
          * 1번 특징이 0인 그룹 ==> 'no' 1개 ==> 정보가 없다
            - 엔트로피: 0
          * 1번 특징이 1인 그룹 ==> 'yes' 2개, 'no' 2개 ==> 확률이 1/2, 1/2
            - 엔트로피: 1.0
          * 1번 특징으로 분할된 두 그룹에 대한 엔트로피의 기댓값
            - 1/5 \* 0.0 + 4/5 * 1.0 = 0.8
            * ==> 0번 특징으로 분할 시 정보 이득이 더 크다
            * ==> 0번 특징이 최선의 분할 특징으로 선택된 것이 일리있음
>###    3.2 매스플롯라이브러리 주석으로 파이썬에서 트리 플롯
        3.2.1 매스플롯라이브러리 주석
            - 매스플롯라이브러리는 애너테이션이라는 도구를 포함하고 있다.
            - 이 애너테이션은 그려진 플롯 내에 있는 데이터 점들을 설명할 수 있도록 주석을 추가하는 도구이다.