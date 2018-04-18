## 파이썬 시작
>###    1. 파이썬 기초
        1. repl : repeat eval print loop -- 데이터 처리 및 print하고 반복하는 환경
        2. idle : IDLE(Integrated DeveLopment Environment) - 통합 개발 환경
            - ipython - gui환경 사용가능
            - jupyter notebook
            - anaconda > 배포판
            - virtual Environment - 금융공학, 머신러닝 등 다른 분야를 해당 분야에 대한 내용만 다루고 싶은 경우 여러 패키지가 아닌 관련 패키지만 설치한 환경을 유지할 수 있게 함.
                - 이를 편안하게 사용하려면 conda 사용
                - conda를 사용치 않은 경우, virtualenv 라이브러리를 설치해 사용해야 함.
            - conda > 아나콘다 설치시 자동 설치
                - package, dependency and environment management for any language로써 필요한 경우 R, java, Scala, C등 다른 언어도 커널을 사용해 환경을 만들 수 있음
                - 환경관리자이며 패키지를 빠르게 설치하고 실행하고 업데이트 해줌
                * 명령어 (conda) > cheat sheet 사용 가능
                    - conda --version : 버젼 확인
                    - conda update conda : 콘다 업데이트
                * managing environment
                    - create and activate an environment
                        conda create --name snowflakes biopython
                        (snowflakes라는 이름의 파이썬 환경을 만들고 biopython패키지 설치)
                    - activate the new environment
                        activate snowflakes
                        (snowflakes를 실행)
                    - create a second environment  
                        conda create -n bunnies python=3.5 astroid babel
                        (다른 환경을 만드는데 버젼을 특정 버젼으로 지정함. astroid와 babel 패키지 설치, bunnies라는 이름의 환경)
                    - List all environments
                        conda into --envs 또는 (환경이름) -- 특정환경의 경우 이름을 주면 됨.
                    - switch to another environment
                        activate bunnies
                        (snowflakes에서 bunnies로 이동)
                    - make an exact copy of an environment
                        conda create -n bunnies python=3.5 astroid babel
                        (기존 파이썬 환경을 복사하여 새로 만듦.)
                    - chaeck to see the exact copy was made
                        conda info --envs
                        conda info -e
                    - delete an environment
                        conda remove --name flowers --all
                    - learn more about environment
                        conda remove --help 또는 conda remove -h
                        
                * managing python : conda는 python도 패키지로 취급
                    - Check Python versions
                        : conda search --full-name python
                        : conda search python
                        (전체 파이썬 버젼 확인)
                    - Install a different version of Python
                        : conda create --name snakes python=2
                        (2버젼 중 default인 (아마 2.7.1)로 snakes설치)
                    - Verify environment added
                    - Verify Python version in new environment
                        : python --version
                    - Use a different version of Python
                        : activate snowflakes
                    - Verify Python version in environment
                        : python --version
                    - Deactivate this environment
                
                * package managing
                    - View a list of packages and versions installed in an environment
                        : activate root
                        : conda list
                          cf.) What data is in the third column of `conda list`
                        : https://github.com/conda/conda/issues/1092
                    - View a list of packages available with the conda install command
                        : https://docs.continuum.io/anaconda/packages/pkg-docs
                    - Search for a package
                        : conda search beautifulsoup4 --name bunnies
                        (bunnies환경에 beautifulsoup4를 찾음)
                    - Install a new package
                        : conda install --name bunnies beautifulsoup4
                        : activate bunnies
                        (기존의 환경에 패키지 추가 설치함)
                    - Install a package from Anaconda.org
                        : go to http://anaconda.org and download the packages
                    - Check to see that the package downloaded
                    - Install a package with pip
                        + pip is ONLY a package manager, NOT environment manager
                        : pip install see
                        (python만 설치시에 패키지 관리할 때 pip로 관리함)
                    - Verify pip installs
                        : conda list
                    - Install commercial package
                        : conda install iopro
                          + free trial expires after 30 days
                * Removing packages, environments, or conda
                    - Remove a package
                        : conda remove --name bunnies iopro
                    - Confirm that program has been removed
                        : conda list
                    - Remove an environment
                        : conda remove --name snakes --all
                        (--all을 주지 않으면 환경 제거가 되지 않음)
                    - Verify environment was removed
                        : conda info --envs
                    - Remove conda
            - Anaconda Navigator(conda의 GUI버젼)
                관리가 매우 편함, 그러나 무겁기 때문에 속도가 느림
                - a desktop graphical user interface (GUI)
                - to launch applications and easily manage conda packages, environments and channels 
                without using command-line commands
                ~ Getting Started
                    : https://docs.continuum.io/anaconda/navigator/getting-started
                - Basic workflow
                - Creating and activating a new environment for a package
                - Finding and installing a package
                - Using an environment
                + To exit Jupyter Notebook:
                    1. Close the notebook tabs or windows in the browser.
                    2. Press Ctrl-C in the terminal window.
                    3. To stop the notebook, in the terminal window, type Y, then press Enter.
                    4. To exit the terminal window, type exit, then press Enter.
              - Jupyter & Jupyter Notebook
                - jupyter notebook명령어로 anaconda command에서 실행할 수 있다.
                - 실행시 실행한 디렉토리를 기반으로 실행된다.
                - 만약 어떤 곳에서 실행해도 특정 디렉토리로 가게 하고 싶은 경우
                    >> jupyter notebook --generate-config 로 .jupyter 설정 파일을 만든다.
                    >> cd .jupyter로 이동
                    >> idle jupyter_notebook_config.py 로 실행
                    >> ctrl+F로 _dir을 찾는다.
                    >> #c.NotebookApp.notebook_dir = '' -- 여기서 '' 사이에 경로를 넣는다.(주석처리 되지 않도록 #은 지운다)
                    
                    : http://jupyter.org/
                    an open source project was born out of the IPython Project in 2014 as it evolved 
                    to support interactive data science and scientific computing across all programming languages
                    ~ Installation
                        : https://jupyter.readthedocs.io/en/latest/install.html
                    - Prerequisite
                        : Python
                    - Using Anaconda and conda
                    - Using pip
                    ~ IPython Documentation & Tutorial
                    + A powerful interactive Python shell
                    + A Jupyter kernel to work with Python code in Jupyter notebooks and other interactive frontends
                        : http://ipython.readthedocs.io/en/stable/
                        : http://ipython.readthedocs.io/en/stable/interactive/tutorial.html
                    ~ Architecture
                          : http://jupyter.readthedocs.io/en/latest/architecture/how_jupyter_ipython_work.html
                    ~ Jupyter Notebook Documentation      
                            : http://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html
                                - Notebook document
                                - Jupyter Notebook App
                                - kernel
                                    + a “computational engine” that executes the code
                                    + the RAM is not released until the kernel is shut-down
                                - Notebook Dashboard
                                : http://jupyter-notebook.readthedocs.io/en/latest/
                    ~ Jupyter Notebook Keyboard Shortcuts
                            : 'H' or 'h' command
                            : https://www.cheatography.com/weidadeyue/cheat-sheets/jupyter-notebook/pdf_bw/
                    ~ Markdown
                            : http://jupyter-notebook.readthedocs.io/en/latest/examples/Notebook/Working%20With%20Markdown%20Cells.html
                    ~ Markdown Cheatsheet
                        : https://github.com/adam-p/markdown-here/wiki/Markdown-Here-Cheatsheet
        3. Zen of Python(파이썬의 철학)
            >>> import this    # Easter Egg
            * 원문 - https://www.python.org/dev/peps/pep-0020/
            * 번역 - http://kenial.tistory.com/903
            * 해설 - http://egloos.zum.com/mcchae/v/10719283
            ~ PEP(Python Enhanced Proposal)
                : 파이썬을 개선하기 위한 개선 제안서
                : PEP Types
                    - Standards Track
                    - Inforamtional
                    - Process
                : https://www.python.org/dev/peps/pep-0001/
            ~ Index of PEP
                : https://www.python.org/dev/peps/
                    - Meta-PEPs(PEPs on PEPs -- 즉, PEPs에 대한 PEPs라는 의미)
                    - Other Informational PEPs
                    - Accepted PEPs (accpeted; may not be implemented yet)
                    - Open PEPs (under consideration)
                    - Finished PEPs (done, implemented in code repository)
                    - ...
            (indentation의 약속, 컬럼 길이에 대한 약속, parameter value indentation의 약속 등 다양한 제안서가 들어있음. 이에 따르는 것이 좋음)
>###    2. 파이썬 특징
        1. 시멘틱
            - 가독성, 명료성, 명백함 강조
            - 공백문자(탭/스페이스)를 이용해 코드 구조화
            - : (colon)을 사용해 entry point 인식
            - ; (semi - colon) 으로 여러 문장을 인식시킬 수 있으나, 한 줄에 하나의 명령문이 원칙
            - 객체 지향
            * 주석
                - # 뒤의 글자는 모두 인터프리터에서 무시됨.
            * 함수
                - 0개 이상의 인자 전달
                - 반환 값 변수 대입 가능
                - 1급 객체로써 취급됨. 
            * 메소드 등등.. 교재참조
>###    3. 프로그래밍
        1. implicit vs explicit
            - automatic - manual
        2. static vs dynamic
            - static : compile-time or before runtime
            - dynamic : runtime
        3. Strictness versus laziness
            - 각종 데이터 전처리(filtering, grouping) 등의 작업을 수행할 때, 필요한 연산을 그 때 그 때 수행하지 않는다.
            - 특정 명령(save, print)들이 수행될 경우에만 수행함.(laziness)
        4. short-cut evaluation
            - 조건문에 or가 있는 경우 전자의 조건의 결과에 따라 후자 조건을 수행하지 않을 수 있다.
        5. first class citizen
            - 변수나 데이타에 할당 할 수 있어야 한다.
            - 객체의 인자로 넘길 수 있어야 한다.
            - 객체의 리턴값으로 리턴 할수 있어야 한다.
            - 컴퓨터 프로그래밍 언어 디자인에서, 특정 언어의 일급 객체 (first-class citizens, 일급 값, 일급 엔티티, 혹은 일급 시민)이라 함은 일반적으로 다른 객체들에 적용 가능한 연산을 모두 지원하는 객체를 가리킨다. 함수에 매개변수로 넘기기, 변수에 대입하기와 같은 연산들이 여기서 말하는 일반적인 연산의 예에 해당한다.
>###    4. 모듈
            - py 파일에서 추가해서 사용할 수 있는 함수, 변수 선언을 담은 .py파일
            - as 예약어로 특정 다른 이름으로 import 가능