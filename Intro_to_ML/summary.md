머신러닝 수업 정리
=================
1. Classification example
    - supervised classification이란?
        - supervised는 수많은 example이 주어지면 해당 example들에 대한 
        합당한 대답을 얻을 수 있다는 의미다.
        - 예시
            - tag된 사진들에서 특정한 사람 찾아내는 것.
            - 특정인의 음악 choice를 통해 특징을 분석하고 추천하는 것.
    - Acerous VS non-Acerous 예제
        - 기린은  Acerous?
        - 그룹으로 지어진 동물들로 미루어 볼 때 각각의
attribute가 다르다. 
        - 기린은 acerous가 아니다.
    - Features and Labels
        - 음악의 feature가 tempo와 intensity로 이루어져 있다고 가정한다.
        - 각 feature의 정도에 따라 호불호에 대한 명확한 label이 지어진다고 가정한다.
        - 언제나 명확한 결과를 예측할 수 있을까?
    - Decision surface
        - 서로 다른 feature를 갖는 object를 나누는 기준
        - 명확히 나눌 수록 좋은 D.S이다.
        - 머신 러닝에서는 DATA를 수집한 다음 그것을 갖고 D.S를 만드는 작업을 한다.
