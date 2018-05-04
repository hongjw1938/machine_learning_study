##  13. pandas를 이용한 데이터 분석 기초
>###    1. pandas Series
        * Series : 1차원 자료구조, DataFrame : 2차원 자료구조
        * 위 내용은 python_class의 lecture 정리 내용 참조
>###    2. pandas DataFrame
        * 위 내용은 python_class의 lecture 정리 내용 참조
>###    3. 주식 데이터 받기
        * pandas-datareader 패키지
            - pandas 0.17.0 버젼부터 독립 패키지가 되었음.
            - 데이터를 읽어오는 데 사용됨
        * DataReader 사용하기
            - DataReader 함수는 웹 상의 데이터를 DataFrame 객체로 만드는 기능 제공
            - DataReader.py참조
        * 차트그리기
            - Chart.py 참조
            - 별도의 창으로 원하는 경우 %matplotlib qt를 사용
            - jupyter notebook의 경우는 %matplotlib inline
>###    4. 이동평균선 구하기
        * 이동평균선 : 일정 기간 동안의 주가를 산술 평균한 값인 주가이동평균을 차례로 연결해 만든 선
        * pandas를 이용한 주가이동평균 계산
            - 일자별 데이터를 DataFrame으로 저장
            - rolling메서드와 mean을 이용해 계산
            - ma5.py참조
        * 주가이동평균선 그리기
            - ma_Chart.py참조
            - 범례의 경우 grid()메서드를 사용하면 표시할 수 있다.
            