##  16장. PyQt를 이용한 GUI 프로그래밍
>###    1. pyqt 기초
        * 설치 및 업데이트
            - conda list pyqt : 패키지 설치 된 목록을 확인할 수 있다.(버젼등)
            - conda update pyqt : 최신버젼으로 업데이트
        * pyqt
            - Widget클래스
                >> 위젯 클래스의 객체를 생성해 UI를 만들 수 있다.
            - 이벤트 루프
                >> QApplication객체에서 exec_ 메서드를 호출해 이벤트 루프 생성
            - 함수, 메서드 구현
                >> 시그널 발생시 호출되는 함수 혹은 메서드 : 슬롯
                >> ex) clicked 시그널 : 클릭되었을 때 발생함
            - pyqt에서는 위젯에서 발생하는 시그널에 대해 어떤 슬롯으로 처리할지에 대해 미리 등록함
            - 특정 위젯에서 시그널이 발생시 이벤트 루프가 미리 연결된 슬롯을 자동으로 호출
            - 이벤트 루프는 사용자가 프로그램 종료 시그널을 보내기 전까지 계속해서 같은 방식으로 슬롯 호출
        * QApplication 객체(qApplication.py 참조)
            - 이벤트 루프는 QApplication클래스의 객체를 생성한 후 exec_ 메서드를 호출하는 순간 생성된다.
            * QLabel : 위젯
        * 위젯의 종류에 따른 시그널
            - 실제 PyQt는 위젯의 종류에 따라 발생 가능한 기본 시그널이 정의되어 있다.
            - ex) QPushButton 위젯은 마우스 클릭을 했을 때, 'clicked'라는 시그널이 발생한다.
            - 슬롯, 즉 함수 또는 메서드를 구현했다면 시그널과 슬롯을 연결해야 한다.
            - 관련 내용은 qPushButton.py에서 참조
            - 해당 내용은 QPushButton위젯에서 clicked이벤트 발생시 clicked_slot함수가 호출되도록 연결해 두었다.
        * 위젯
            - PyQt에서는 모든 위젯이 최상위 위젯을 의미하는 윈도우가 될 수 있다.
            - 그러나 대부분 프로그램에서 QMainWindow나 QDialog 클래스를 사용해 윈도우를 생성한다.
            - makeWindow.py 코드 참조
>###    2. Qt Designer
        * Qt Designer
            - Qt의 컴포넌트를 이용해 GUI를 설계하는 전용 툴
            - 위지위그방식 : WYSIWYG
                >> What You See Is What You Get
        * 구성
            1) GUI 레이아웃
            2) 시그널 - 슬롯 연결 및 슬롯 처리 함수(메서드)
            3) 이벤트 루프
            - 이 중 이벤트 발생시 수행할 함수를 작성하는 작업은 프로그래머의 작업임
            - 그러나 버튼 등의 위젯을 생성하고 화면에 출력될 윈도우의 크기를 조절하는 것과 같은 사소한 작업은 툴을 이용함.
            - C:\Anaconda3\Library\bin 디렉터리에는 designer.exe파일이 존재함
>###    3. 기본 위젯
        ** cf) self
            - 생성 위젯을 클래스 내의 다른 메서드에서 참조시에는 self를 붙인다.
            - 아닌 경우에는 붙이지 않아도 된다.
        * QPushButton
            - 버튼을 생성하기 위해 사용하는 위젯
            - 사용자로부터 "예", "아니오" 같은 이벤트를 받는 데 사용
            - widget_QPushButton.py 참조
            - QCoreApplication.instance()를 이용시 app변수가 바인딩하고 있는 동일 객체를 얻어올 수 있다.
            - app변수가 바인딩하고 있는 객체는 QApplication클래스의 인스턴스이다. 해당 객체는 quit메서드를 제공한다.
            - quit메서드가 호출되면 윈도우가 종료되는 것이다.
            - widget_QPushButton2.py에 나오듯이 app을 전역변수로 사용하는 것은 유지보수 측면에서 좋지 않다.
        * QLabel
            - 텍스트 혹은 이미지를 출력할 때 사용됨.
            - widget_QLabel.py를 참조
            - move 메서드는 부모 객체에서 이동하는 정도를 지정
            - resize 메서드로 크기를 조정
            - setText, clear메서드를 통해 Text를 지정하거나 지울 수 있다.
        * QLineEdit, QStatusBar
            * QLineEdit
                - 한 줄의 텍스트를 입력할 수 있는 위젯
                - 사용자로부터 간단한 텍스트를 입력받을 때 사용함.
                - widget_QLineEdit.py 참조
                - 사용자로부터 코드를 QLineEdit을 이용해 입력받을 수 있다.
                - 적당한 시그널을 이용해 사용자 이벤트를 처리해야 한다.(textChanged)
                * QLineEdit 시그널
                    1) textChanged() : QLineEdit객체에서 텍스트가 변경될 때
                    2) returnPressed() : QLineEdit객체에서 사용자가 엔터 키를 눌렀을 떄
                - 위 py코드에서는 QStatusBar를 사용해 위젯을 생성하고 사용자에게 보여주도록 하였다.
        * QRadioButton, QGroupBox
            * QRadioButton : 사용자로부터 여러 옵션 중 하나를 입력받을 때 사용
                - QRadioButton생성자를 호출해 객체 생성 가능
                - 첫 인자 : 출력할 문자열 / 두 번째 인자 : 출력될 부모 위젯(parent widget)
                - setChecked메서드 : 버튼의 초기 상태 설정
            * QGroupBox : 제목이 있는 네모 박스 형태의 경계선 만드는 데 사용
                - 네모 박스를 만듦
                - resize메서드는 너비, 높이를 인자로 받는다.
        * QCheckBox
            - 여러 옵션 동시 선택 가능
            - widget_QCheckBox.py참조
            - isChecked메서드를 이용해 선택 여부를 확인할 수 있다.
        * QSpinBox
            - 사용자로부터 정수값을 입력받을 때 사용.
            - 화살표는 값을 증가 혹은 감소시킬 때 사용한다.
            - 생성자를 이용해 객체를 생성할 수 있으며, 텍스트를 출력하지 않으므로 부모 위젯만 인자로 전달
            - widget_QSpinBox.py를 참조
            - valueChanged시그널을 이용해 값이 변경시 사용자 이벤트에 대응할 함수를 지정할 수 있다.
            * 유용한 메서드
                1. 초깃값 설정 : setValue()
                2. 증감 수준 설정 : setSingleStep()
                3. 값의 범위 설정 : setMinimum(), setMaximum()
        * QTableWidget
            - 2차원 포맷 형태의 데이터를 표현
            - 행과 열의 개수 지정 : setRowCount(), setColumnCount()
            - setItem메서드 : 행, 열 인덱스와 QTableWidgetItem객체를 전달받는다.
            - widget_QTableWidget.py와 -2.py참조
            - setEditTriggers메서드 : 아이템 항목을 사용자가 수정할 수 없게 만들 수 있다.
            - row방향 라벨 설정 : setVerticalHearderLabels메서드 사용
            - column방향 라벨 설정 : setHorizontalHeaderLabels메서드 사용
            - Qt.AlignVCenter : 수직적으로 center를 기준으로 정렬함.(Top, Bottom도 있음. 위쪽, 아래쪽으로 정렬함.)
            - Qt.AlignRight : 우측 정렬
            - resizeColumnsToContents, resizeRowsToContents : 각 행렬을 아이템 길이에 맞춰 조정한다.
>###    4. Layout
        - 이전 코드에서 이미 move, resize, setGeometry를 통해 여러 기능을 사용함
        - setGeometry메서드는 위젯의 출력 위치를 결정한다.
        - layout_16.py 참조
            >> QWidget를 상속하여 UI를 생성함.
            - 위젯의 크기와 출력 위치를 명시적으로 설정하는 것은 윈도우 크기 변경시 문제가 발생한다.
            - PyQt는 위의 문제를 해결코자 레이아웃 매니저를 제공한다.
            - 그 매니저는 QVBoxLayout, QHBoxLayout, QBoxLayout, QGridLayout, QLayout이 있다.
            - 레이아웃 매니저에 추가할 위젯 생성시에는 부모 위젯을 지정하지 않으며, 위치, 크기를 명시하지 않는다.
        * QVBoxLayout
            - 위젯을 수직방향으로 나열
            - layout_QVBoxLayout.py 참조
            - 일정 비율을 가지며 내부 위젯이 크기가 자동으로 바뀜
        * QHBoxLayout
            - 행 방향으로 위젯을 배치할 때 사용하는 레이아웃 매니저
            - layout_QHBoxLayout.py 참조
        * QGridLayout
            - 격자 형태의 UI를 구성할 때 사용한다.
            - 위젯을 입력할 좌표를 입력받음.
            - layout_QGridLayout.py 참조
        * 레이아웃 중첩
            - layout_overlap.py 참조
>###    5. 다이얼로그
        - 사용자와의 상호작용을 위해 사용되는 윈도우
        * QFileDialog
            - 사용자가 파일이나 디렉터리를 선택할 수 있게 하는 다이얼로그 창
            - 파일 선택시 해당 파일의 절대 경로가 윈도우에 출력된다.
            - dialog_QFileDialog.py참조
            - 반환된 파일 경로는 튜플 타입으로 fname이라는 변수가 바인딩한다.
        * QInputDialog
            - 사용자로부터 간단한 텍스트, 정수, 실수를 받을 때 사용함.
            - ok와 cancel버튼이 존재함.
            - dialog_QInputDialog.py 참조
            - getInt메서드 : 부모 위젯, 창에 표시할 텍스트, 창 내부에 출력할 텍스트를 인자로 받는다.
            - getInt메서드는 (text, ok)의 튜플 형태로 값을 반환한다.
                >> text에는 적어넣은 값, ok에는 ok를 누른 경우 True가 반환된다.
            * 이외의 메서드
                - getDouble
                - getText
                - getItem
            - getItem을 사용하는 경우 : dialog_QInputDialog2.py를 참조
                >> 인자
                    1. 부모 위젯
                    2. 타이틀 텍스트
                    3. 내부 텍스트
                    4. 선택할 아이템 리스트
                    5. 초기 아이템 인덱스
                    6. 아이템 수정 가능 여부
        * 메인 윈도우와 다이얼로그의 상호작용
            - setWindowIcon메서드 : 타이틀에 출력할 이미지 파일을 설정
            - dialog_window_dialog.py참조
>###    6. PyQt와 matplotlib 연동
        - PyQt위젯, 그래프 동시 배치 하기
        * 기본 레이아웃 구성
            - matplotlib_base_layout.py참조
            - matplotlib을 이용해 PyQt 내에 그래프를 그릴 경우 FigureCanvasQTAgg 클래스를 사용해야 한다.
            - addStretch메서드를 통해 객체를 상단 배치하고 크기 조절 가능한 공백을 추가한다.
            - setStretchFactor의 2번째 인자를 0(False)로 하면 크기 조절이 불가하다. 1로 하면 가능
        * 그래프 그리기
            - matplotlib_Drawing_Graph.py참조
            - 현재 pandas_datareader클래스가 deprecated된 상태
            - http://excelsior-cjh.tistory.com/109 해당 내용 참조
            
            - matplotlib_Drawing_Graph2.py참조
            - df['MA20'] = df['close'].rolling(window=20).mean()
            - df['MA60'] = df['close'].rolling(window=60).mean()
            - 위 코드는 pandas의 DataFrame의 rolling함수
                * https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.rolling.html참조
                >> rolling함수
                    - Provides rolling window calculations.
                    - 인자로 1번째는 window의 크기를 받는다. : 데이터의 개수. 20이면 20개씩
                    - 즉, 위의 코드는 20일치, 60일치의 데이터를 받아서 20일, 60일 이동평균을 구할 수 있다.
                    
            - datareader_test.py를 참조하면 datareader가 데이터를 읽고, 관련 데이터 프레임을 확인할 수 있다.