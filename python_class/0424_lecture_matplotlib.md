## 도식화와 시각화
>###    * matplotlib(matlab, matplotlib 사이트참조) 
        - 모든 것을 행렬, 즉 matrix로 처리함.
        - 스칼라도 1*1 행렬
        - 2D 도표를 위한 데스크톱 패키지로, 출판물 수준의 도표를 만들 수 있도록 설계됨
        - Ipython에서 GUI툴킷과 함께 matplotlib을 사용시 도표의 확대와 회전 같은 인터렉티브한 기능 ㅏ용 가능
        - 도표 저장도 가능
        - 일반적으로 plt라고 줄여서 import함. 규칙.
>###    8.1 matplotlib API간략하게 살펴보기
        1. Figure와 서브플롯
            - 그래프를 위한 객체 figure
            - figsize : 그래프가 디스크에 저장될 경우 크기나 가로세로 비율 결정
            - 객체는 숫자를 인자로 받는다.
            - 활성화된 figure객체는 plt.gcf()로 참조할 수 있다.
            - 빈 figure객체로는 그래프를 만들 수 없어, add_subplot을 사용해 최소한 하나 이상의 서브플롯을 생성해야 한다.
            - ax1 = fig.add_subplot(2, 2, 1) #객체는 크기가 2 * 2이고 4개의 서브플롯 중에서 첫 번째를 선택하겠다는 의미
            - 서브플롯은 1부터 숫자가 매겨진다.
            - 서브플롯은 2차원 배열로 쉽게 색인될 수 이다.
            - subplots함수를 통해 fig객체와 axes객체를 한 번에 반환해서 사용할 수 있다.
            - fig객체는 figure전체, axes는 각각의 서브플롯
        2. 색상, 마커, 선 스타일
            - plot함수는 X와 Y 좌표 값이 담긴 배열과 추가적으로 색상과 선 스타일을 나타내는 축약 문자열을 인자로 받음
            - ex) ax.plot(x, y, 'g--')
            - linestyle은 plot메서드를 참조
            * matplotlib스타일
                - matplotlib.style.available, matplotlib.style.library 로 확인가능
                - dir(matplotlib.style)
            * ggplot 스타일 : matplotlib.style.use('ggplot')
                - gcf()메서드 사용
            * seaborn 스타일 : matplotlib.style.use('seaborn')
                #seaborn style
                    ax2 = fig.add_subplot(222)
                    plt.plot(np.random.randn(30).cumsum(), 'b.') #점으로 그림.
                - anaconda prompt에서 실행해보면 격자의 색이 다른 것을 알 수 있다.
            * 원래의 스타일 : matplotlib.style.use('default')
        3. 눈금, 라벨, 범례
            - xlim, xticks, xtickslabels 같은 메서드로 표의 범위를 지정하거나, 눈금의 위치, 눈금의 이름을 각각 조절할 수 있다.
            - 아무런 인자가 없이 호출시 현재 설정된 매개변수의 값을 반환, plt.xlim은 현재 x축의 범위 반환
            - 인자 전달시 매개변수의 값을 설정, plt.xlim([0, 10])을 호출시 X축의 범위가 0부터 10까지
        4. 주석과 그림 추가
            - text, arrow, annotate함수를 이용해 추가할 수 있다.
        5. 그래프를 파일로 저장
            - plt.savefig메서드로 파일 저장 가능.
        6. matplotlib 설정
            - 도표크기, 서브플롯 간격, 색상, 글자 크기, 격자 스타일과 같은 설정이 가능
            - rc메서드를 활용해 프로그래밍 적으로 설정 가능
                - ex) plt.rc('figure', figsize=(10, 10))
            - 또는 matplotlib/mpl-data디렉터리에서 matplotlibrc파일에 저장된 설정과 옵션의 종류를
            - 수정하여 .matplotlibrc라는 이름으로 사용자 홈 디렉토리에 저장시 matplotlib을 불러올 때마다 사용가능
>###    8.2 pandas에서 그래프 그리기
        - pandas를 이용해 다양한 온전한 그래프를 그리기 위해 필요한 많은 matplotlib코드를 간단하게 표현할 수 있다.
        1. 선 그래프
            - plot 메서드를 통해 다양하게 그릴 수 있다.
        2. 막대 그래프
            - kind='bar' 옵션 혹은 kind='barh'(수평 막대) 로 막대그래프를 그릴 수 있다
            - Series나 DataFrame의 색인은 X, Y 눈금으로 사용됨.
        3. 히스토그램과 밀도 그래프
            - 값의 빈도를 분리해 출력
        4. 산포도
            - 2개의 1차원 데이터 묶음 간의 관계를 나타내고자 할 때 유용한 그래프다.
            - scatter메서드를 사용해 산포도를 그릴 수 있다.
>###    8.3 아이티 지진 데이터 시각화
        
