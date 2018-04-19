##Numpy sorting
import numpy as np

arr = np.random.randn(8)
arr
arr.sort()
arr

arr = np.random.randn(8)
np.sort(arr) 
arr #원본은 그대로

arr = np.random.randn(5, 3) #2차원 배열은 정렬 축을 고려해야함. 행 / 열로 정렬 방식 고려, axis인자를 주면 됨.
arr
arr.sort(1)    # 0으로 주면 열을 정렬, 1은 행을 정렬
arr


large_arr = np.random.randn(1000)
large_arr.sort()
large_arr

large_arr[int(0.05 * len(large_arr))] # 5% quantile, 즉, 정렬 이후 첫 5%에 해당하는 분위수 값 반환


##집합 함수

names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.unique(names) #중복값 없애고 정렬해서 리턴

ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])
np.unique(ints)


sorted(set(names)) #np 모듈을 사용하지 않고 기본 파이썬 내장 함수를 사용하는 경우

#불리언 배열 반환
values = np.array([6, 0, 0, 3, 2, 5, 6])
np.in1d(values, [2, 3, 6])


## 배열의 파일 입출력
arr = np.arange(10)
np.save('some_array', arr) #저장

np.load('some_array.npy') #로드, 단순히 메모장으로 열면 전부 인식할 수 없음

np.savez('array_archive.npz', a=arr, b=arr) #압축된 형식으로 저장됨.


##선형대수

x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
x
y
x.dot(y)  # equivalently np.dot(x, y), 행렬의 곱셈을 구하는 함수

np.ones(3) # 1행렬로 3 * 1 형식으로 반환
np.dot(x, np.ones(3))


from numpy.linalg import inv, qr # numpy.linalg는 행렬의 분할, 역행렬, 행렬식 등을 포함함.
X = np.random.randn(5, 5)
mat = X.T.dot(X) #T는 전치행렬, 전치행렬과 X를 곱한 것.
mat

inv(mat) #역행렬
mat.dot(inv(mat)) # 값이 정확히 단위행렬이 이닌 것처럼 보이는데, 이는 정확도에 의해서 그러함. e-16과 같은 경우 매우 작은 값이므로 무시 가능
                  # 기존 행렬과 역행렬을 곱했으니 단위행렬이 나오는 것이 정상.

q, r = qr(mat) #하나의 행렬을 q, r의 두 가지 행렬로 나타낼 때 사용
r


##난수 생성
samples = np.random.normal(size=(4, 4))
samples

from random import normalvariate
N = 1000000  #백만번 반복
%timeit samples = [normalvariate(0, 1) for _ in range(N)] # %timeit을 이용해 백만개의 랜덤 난수를 구하는데 각각의 loop에서 시간이 어느 정도 걸렸는지 확인가능
%timeit np.random.normal(size=N)                          # 10개의 루프에 대해 평균을 자동으로 구해줌

##계단오르기 예제

#np모듈을 사용하지 않았을 때.
import random
position = 0
walk = [position]
steps = 1000
for i in range(steps):# i값이 어차피 쓰이지 않으니까 i 대신에 _를 써서 의도적으로 사용하지 않는다고 명시 가능
    step = 1 if random.randint(0, 1) else -1 #randint는 0혹은 1을 반환 여기서 0이 true
    position += step
    walk.append(position) #walk의 리스트를 통해 계단 이동현황을 trace함.
walk

%matplotlib inline
import matplotlib.pyplot as plt

plt.plot(walk[:100])


steps = 1000
s1 = [1 if random.randint(0, 1) else -1 for _ in range(steps)]
walk = [sum(s1[:x]) for x in range(len(s1)+1)]
walk[:20]

s1[:10]

plt.plot(walk[:1000])

#np모듈을 사용했을 때

nsteps = 1000
draws = np.random.randint(0, 2, size=nsteps) #random을 미리 1000개 만들어버림. 내장 randint는 반복 기능 없지만 np모듈에는 있음.
steps = np.where(draws > 0, 1, -1)           #step도 미리 정해놓음
walk = np.r_[[0], steps.cumsum()]            #정해진 조건에 따라 누적 더하기, np.r_는 0부터 시작하라는 의미
walk.min()
walk.max()

np.abs(walk) >= 10 # bool 배열 반환


(np.abs(walk) >= 10).argmax()  #시작위치에서 10계단 이상 올라가거나 내려간 경우의 수를 보여줌
                               #argmax()는 그 중 최대값이 나온 경우의 수를 반환, 제일 처음 나온 최대값이 26번째에 나왔다는 의미이다.
                               
                               
                               
nwalks = 5000   #계단 오르내리기 5000번 시도
nsteps = 1000

draws = np.random.randint(0, 2, size=(nwalks, nsteps)) # 크기가 2인 튜플을 넘기면 2차원 배열 생성
draws.shape
draws[:5, :10]

# draw값이 0이면 한 계단 내려가기, 1이면 한 계단 올라가기
steps = np.where(draws > 0, 1, -1)
steps[:5, :10]

walks = steps.cumsum(1) #가로로 더해줌.
walks[:5, :10]


#hstack 예시
a = np.array((1, 2, 3))
b = np.array((2, 3, 4))
np.hstack((a, b))

a = np.array([[1],[2],[3]])
b = np.array([[2],[3],[4]])
np.hstack((a,b))


walks = np.hstack((np.zeros((nwalks, 1), dtype=np.int32), walks)) #np모듈을 사용했을 때. hstack으로 더해줌.
walks[:, :4]  #앞에 0이 추가되어있는 것을 확인 할 수 있음.

walks.max()
walks.min()

hits30 = (np.abs(walks) >= 30).any(1)
hits30
hits30.sum() # 30 혹은 -30에 도달한 시뮬레이션의 개수
             # 첫 째 값이 false라는 것은 첫 1000번의 시도에서 한 번도 30회 이상 위 아래로 간 적이 없었다는 의미이다.
             
crossing_times = (np.abs(walks[hits30]) >= 30).argmax(1) #argmax(1)에서 1은 축의 번호, 최대값이 나온 원소의 위치값(최초 30 계단)
crossing_times
crossing_times.shape

crossing_times.mean() # 그 30계단 이상 갔던 경우의 행렬 중에서 30계단 이상 갔던 최초의 원소값의 위치의 평균

steps = np.random.normal(loc=0, scale=0.25,
                         size=(nwalks, nsteps)) #randint가 아닌 정규분포를 따르는 경우에서 난수 발생시킴.
steps