##  10장. k-평균 군집화 : 항목 표시가 없는 아이템 그룹 짓기
>###    10.0 군집화
        * 분류와의 차이점
            - 분류는 미리 찾고자 하는 것이 무엇인지 인지하고 있음
            - 군집화는 그렇지 않다.(비지도 분류라고도 한다.)
        * k-평균 알고리즘
            - k개의 서로 다른 군집을 찾는다.
            - 각 군집의 중심이 이 군집 내에 있는 값들의 평균.
>###    10.1 k-평균 군집화 알고리즘
        * k-평균 군집화
            - 장점 : 구현이 쉽다.
            - 단점 : 지역 최소점에 수렴될 수 있다. 데이터 집합이 매우 큰 경우 처리 시간이 오래 걸린다.
            - 활용 : 수치형 값.
            - k-평균은 주어진 데이터 집합에서 k개의 군집을 찾고자 하는 알고리즘이다.
            - 군집의 개수인 k는 사용자가 정의한다.
            - 각 군집은 중심(centroid)라고 하는 하나의 단일한 점으로 묘사된다.
            - 중심은 군집 내에 있는 모든 점들의 중심을 의미한다.
            * 순서
                - k개의 중심은 각각 하나의 점이 되도록 임의로 할당한다.
                - 데이터 집합에 있는 점들을 각각 하나의 군집에 할당
                - 할당은 데이터 점에서 가장 가까운 중심을 찾아 이에 해당하는 군집으로 데이터 점을 할당함.
                    (즉, 중심점과 데이터 집합의 각각의 거리를 계산해서 가까운 곳으로 할당)
                - 각 군집 내에 있는 모든 점에 대한 평균값으로 각 군집의 중심을 모두 갱신
        * k-means를 도와주는 함수
            * loadDataSet()함수
                - 부동 소수점 데이터를 탭으로 구분한 텍스트 파일을 하나의 리스트로 불러옴.
                - 각각의 리스트에는 dataMat이라고 하는 리스트가 추가됨.
                - 반환 값은 다른 많은 리스트들을 포함하고 있는 하나의 리스트이다.
            * distEclud()함수
                - 두 값 간의 유클리드 거리를 계산한다.
            * randCent()함수
                - k개의 중심을 원소로 하는 하나의 집합 생성
                - 초기에는 중심집합을 0행렬로 초기화
                - 임의로 중심점을 선택한 값은 데이터 집합 범위 내에 있어야만 한다.
                - 따라서, 중심점이 데이터 범위 내에 있으면서 0~1.0 사이의 값이 되도록 범위와 최솟값으로 크기를 변경 
        * k-means 알고리즘
            * kMeans()함수
                - k개의 중심을 생성.
                - 각 점을 가장 가까운 중심에 할당
                - 중심 재계산
                - 데이터 점이 군집을 변경하지 않을 때까지 반복된다.
>###    10.2 후처리로 군집 성능 개선하기
        - 행렬이 가지는 군집 할당은 각각의 점에 오류를 나타내는 값
        * 지역 최소점에 수렴하여 군집할당이 좋지 않은 경우
            - p269참조.
            - 최선은 전체 최소점에 수렴해야 하나, 지역 최소점에 수렴할 수 있다.
            - 따라서 SSE(오류 제곱의 합)을 이용해 이 값이 더 낮은 경우 군집화가 더 잘 이루어졌다는 판단을 한다.
            - 그러나 군집화의 질을 높이기 위해 군집의 수를 늘리는 것은 옳지 못하다.
            - SSE가 가장 높은 군집을 두 개의 군집으로 분할한다.(군집 후처리)
            - 큰 군집에 있는 점들을 골라낸 후, k 값을 2로 설정한 다음 k-평균을 수행해 분할