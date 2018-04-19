## 고급 Numpy

from __future__ import division
from numpy.random import randn
from pandas import Series
import numpy as np
np.set_printoptions(precision=4)
import sys

##배열 재형성

arr = np.arange(8)
arr
arr.ndim

arr.reshape((4, 2))
arr.reshape((4, 2)).ndim # 2차원


arr = np.arange(15)
arr.reshape((5, -1)) #2차원, -1을 넣게 되면 행에 맞는 형태로 만듦.
arr.reshape((1, 5, -1)) #3차원

other_arr = np.ones((3, 5))
other_arr.shape
arr.reshape(other_arr.shape) # 특정 배열을 넣게 되면 parameter로 받은 배열의 크기로 reshape

#평탄화

arr = np.arange(15).reshape((5, 3))
arr
arr.ravel() #데이터 복사본 생성하지 않음, 원본 그대로.

arr.flatten() #차원을 낮추어서 풀어버림, 항상 데이터의 복사본 반환


##C와 Fortran순서
arr = np.arange(12).reshape((3, 4))
arr
arr.ravel()
arr.ravel('F') #값을 읽는 방식

##배열 이어붙이고 나누기

arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[7, 8, 9], [10, 11, 12]])
np.concatenate([arr1, arr2], axis=0) #concatenate함수로 연결. 연결 객체를 리스트로 묶어서(튜플도 가능) axis의 기준에 따라 결합
np.concatenate([arr1, arr2], axis=1)

np.vstack((arr1, arr2))
np.hstack((arr1, arr2))


#split
first, second, third = np.split(arr, [1, 3]) #나누는 기준을 정함. row1과 3을 기준으로 나누는 것. 기준이 2이므로 3개가 됨.
first
second
third

#배열쌓기
arr = np.arange(6)
arr1 = arr.reshape((3, 2))
arr2 = randn(3, 2)
arr1
arr2

np.r_[arr1, arr2]  #row 단위로 arr1, arr2를 붙여서 쌓음
np.c_[np.r_[arr1, arr2], arr] #arr1, arr2를 row로 붙인 것에 arr을 추가로 column 단위로 쌓음


np.c_[1:6, -10:-5]


###BroadCasting


arr = np.arange(5)
arr
arr * 4

arr = randn(4, 3)
arr.mean(0)
np.mean(arr) #전체 평균
np.mean(arr, axis=0)

demeaned = arr - arr.mean(0)
demeaned # 4 * 3의 크기
demeaned.mean(0) # 크기3 인 1차원, 2차원으로 만들면 1 * 3, 그러고 크기를 맞추어준다.

arr   # 4 * 3 크기의 2차원 배열
row_means = arr.mean(1)  # 1차원 배열 크기는 4
row_means


row_means.reshape((4, 1))  # 4 * 1의 크기로 변경, 왜냐면 브로드캐스팅 규칙으로 한다고 해도 4 * 4가 되니까 뺄셈이 진행될 수 없기 때문이다.

demeaned = arr - row_means.reshape((4, 1)) #브로드 캐스팅에 의해 4 * 3으로 크기가 변경되고 그에 맞게 계산됨.
demeaned

demeaned.mean(1)

arr
arr - arr.mean(1) # arr은 4 * 3인데 arr.mean(1)의 크기가 단순히 4이니까 차원 변경 후 크기 맞추어도 알맞지 않음

#따라서 아래와 같이 reshape해서 크기를 변경시켜준 다음에 broadcasting 규칙에 따라야 함.
arr - arr.mean(1).reshape((4, 1))


arr = np.zeros((4, 4))
arr_3d = arr[:, np.newaxis, :] # np.newaxis는 새로운 축을 추가하는 것. 중간에 추가되었음. 그래서 4 * 1 * 4

arr_3d
arr_3d.shape

arr[:, :, np.newaxis]
arr[:, :, np.newaxis].shapes


arr_1d = np.random.normal(size=3) # 표준정규분포에서 크기 3의 난수 생성
arr_1d[:, np.newaxis] # 새로 추가되는 축을 1번 축에 두고 기존은 0에 둠.
arr_1d[np.newaxis, :] # 새로 추가되는 축이 0번 축이고 기존은 1


arr = randn(3, 4, 5) # 3차원 배열인 크기는 3 * 4 * 5로 배열 하나 생성
depth_means = arr.mean(2) # 2번축을 기준으로 평균을 구함. 그래서 depth_means는 차원이 3 * 4가 되는 것이다.
depth_means

row_means = arr.mean(1) # 1번축을 기준으로 평균을 구함. 그래서 3 * 5 가 된다.
row_means

demeaned = arr - depth_means[:, :, np.newaxis] #arr은 3 * 4 * 5, 새로 추가한 것은 1, 따라서 3 * 4 * 1 / 이를 브로드캐스팅
demeaned.mean(2) #집계를 하니까 다시 3 * 4의 2차원 배열이 남게 된다.


#브로드캐스팅 이용해 배열에 값 대입하기.

arr = np.zeros((4, 3))
arr[:] = 5
arr # 4 * 3의 크기의 2차원 배열

col = np.array([1.28, -0.42, 0.44, 1.6])
col # 크기 4, 브로드캐스팅으로 arr과 연산이 불가함.

#따라서 차원 하나 추가시킴.
arr[:] = col[:, np.newaxis] #col의 경우 축이 하나 추가되어 기존의 크기 4에서 4 * 1의 2차원 배열이 된다. 여기서 브로드캐스팅하면 됨.
arr

arr[:2] = [[-1.37], [0.509]] # arr[:2] == arr[:2, :]이므로 크기는 2 * 3의 2차원 배열이다. 우측은 중첩 리스트. ndarray로 자동으로 바뀜.
arr[:2]                      # 그래서 2 * 1 매트릭스가 되는 것. 브로드 캐스팅에 의해 크기 변경되고 값이 들어감.ㄴ

arr