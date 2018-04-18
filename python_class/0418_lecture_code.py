#Should install numpy module through conda prompt or navigator
import numpy as np

data = np.random.randn(2, 3)    #Get matrix 2 by 3 filled with random number
data


data
data * 10
data + data

type(data)  #Entity of ndarray within numpy

data.shape  #배열의 dimension과 item은 shape 속성으로 결정됨.
data.dtype

data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)  #리스트의 각각의 원소를 인자로 1차원 배열 생성
arr1

type(arr1) #numpy의 ndarray객체


data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)  #2차원 배열 생성
arr2
arr2.ndim   # ndarray객체가 몇 차원 배열인지 확인
arr2.shape  # n차원 배열의 행렬 사이즈 리턴

arr1.dtype  # float 타입, 64비트
arr2.dtype  # int 타입, 32비트, 만약 타입을 변경하고 싶으면 만들 때 타입을 지정하면 됨.


np.zeros(10) # 0행렬 생성, 넘겨받는 인자의 갯수 만큼.
np.zeros((3, 6))
np.empty((2, 3, 2)) #행렬의 원소 값을 garbage값으로 즉, 메모리에 있는 초기화되지 않은 아무런 값으로 생성

#zeros_like, ones_like함수는 주어진 배열과 동일한 모양과 dtype을 가지는 배열을 0 혹은 1로 초기화해서 생성하는 것.
np.ones_like(arr2)
np.zeros_like(arr2)


np.arange(15)  #파이썬의 range와 유사함. arange는 array range의 의미

np.arange(20, step=3)
np.arange(20, step=3, dtype=np.float32) #default 데이터 타입이 아닌 방식으로 타입을 지정했음.

np.eye(3) #단위 행렬 생성
x = np.empty((4, 4))
x


arr1 = np.array([1, 2, 3], dtype=np.float64)
arr2 = np.array([1, 2, 3], dtype=np.int32)
arr1.dtype
arr2.dtype

arr = np.array([1, 2, 3, 4, 5])
arr.dtype
float_arr = arr.astype(np.float64) # 데이터 타입 변경, 64비트 실수값으로 하는데, arr이 바뀌지 않음. 변환한 것을 리턴만 함.
float_arr.dtype


arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])
arr
arr.astype(np.int32)    # 이 경우 float가 int가 되었기 때문에 소수점이 버려짐/.


numeric_strings = np.array(['1.25', '-9.6', '42'], dtype=np.string_) #' ' 때문에 문자 데이터임. np.string_ 대신 S로도 가능
numeric_strings.astype(float) #또는 numeric_strings.astype(np.float32)

int_array = np.arange(10)
calibers = np.array([.22, .270, .357, .380, .44, .50], dtype=np.float64)
int_array.astype(calibers.dtype) # calibers 배열과 같은 타입으로 변경하는 것.

empty_uint32 = np.empty(8, dtype='u4') #부호가 없는 32비트 정수형
empty_uint32


#배열과 스칼라 간의 연산
arr = np.array([[1., 2., 3.], [4., 5., 6.]])
arr

arr * arr
arr - arr

1 / arr # broadcasting 규칙 적용. arr과 같은 크기의 2차원 배열처럼 변화함. 이것이 broadcasting
arr ** 0.5

#indexing and slicing

arr = np.arange(10)
arr
arr[5]
arr[5:8]
arr[5:8] = 12 # 이 경우도 브로드 캐스팅에 의해 길이 변경됨. list의 경우는 이와 같은 코드 작성 시 에러 발생
arr

l = list(arr)
l
#l[5:7] = 11 #에러발생
l[5:7]

arr_slice = arr[5:8]
arr_slice[1] = 12345
arr
arr_slice[:] = 64
arr  # numpy의 배열은 slicing하면 원본 그대로를 참조한 상태에서 일부만 가져오기 때문에 원본 데이터 변경
     # numpy는 원래 다량의 데이터 처리를 가정하고 만든 패키지라서 객체를 복사하면 많은 메모리 자원을 사용하게 될 가능성이 있기 때문이다.

p_list = [1, 2, 3, 4, 5]
p = p_list[0:4]
p[0] = 300
p
p_list #원본은 변경되지 않았음을 확인할 수 있다.

arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # open bracket이 2개 이므로 2차원 배열
arr2d
arr2d[2]

#배열 값을 참조할 때 아래의 두 가지 중 아무 방법을 사용해도 무관
arr2d[0][2]
arr2d[0, 2]


#3차원 배열
arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d

arr3d.ndim
arr3d.shape #결과가 2, 2, 3인 것을 알 수 있는데 차원이 늘어나면 추가된 차원이 앞 쪽에 붙기 때문이다.

#1차원 배열
a1 = np.array([1, 2, 3])
a1
a1.ndim, a1.shape # (3, )로 나오는 것은 원소가 한 개인 튜플이기 때문임.

#2차원 배열
a2 = np.array([[1, 2, 3], [4, 5, 6]])
a2
a2.ndim, a2.shape

#1차원 배열처럼 보이는 2차원 배열(주의!)
a21 = np.array([[1, 2, 3]])
a21
a21.ndim, a21.shape

arr3d[0]
#index는 축, 로우, 컬럼 순서

old_values = arr3d[0].copy() #명시적 복사
arr3d[0] = 42
arr3d
arr3d[0] = old_values
arr3d


arr2d
arr2d.shape
arr2d[:2]

arr2d[:2, 1:] # 행은 0, 1행, 열은 1열부터 끝까지

arr2d[1, :2]
arr2d[2, :1]

arr2d[:, :1]

arr2d[:2, 1:] = 0


#Boolean indexing
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
names
data

names == 'Bob' #Bob은 스칼라이기 때문에 브로드 캐스팅 이루어짐.

data[names == 'Bob'] # 색인하고자 하는 값이 해당 위치에서 true, false로 리턴됨.

#true인 것 중 일부만 색인
data[names == 'Bob', 2:]
data[names == 'Bob', 3] #하나만 지정했기 때문에 차원이 감소


names != 'Bob'
data[~(names == 'Bob')] #물결표시는 tilde라고 부르는데 not의 역할을 대신해준다.


mask = (names == 'Bob') | (names == 'Will') # | 는 or의 의미이다.
mask
data[mask]

data[data < 0] = 0 # 0보다 작은 값들은 전부 0 으로 바꿈.
data

data[names != 'Joe'] = 7 # 'Joe'가 아니면 7로 바꿈. 당연히 내부적으로 모두 for를 돌려서 반환함.
data


##fancy indexing
arr = np.empty((8, 4))
for i in range(8):
    arr[i] = i
arr


arr[[4, 3, 0, 6]] # 4, 3, 0, 6번째 순서로 4개의 로우를 색인
arr[[-3, -5, -7]] #음수 정수 배열로 색인하면 뒤에서부터

# more on reshape in Chapter 12
arr = np.arange(32).reshape((8, 4)) # 0 ~ 31까지 정수를 만들고 8 * 4의 배열로 생성
arr
arr[[1, 5, 7, 2], [0, 3, 1, 2]]

arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]] # 특정 행의 순서로 해당 행 전체의 값을 특정 열의 순서로 출력

arr[np.ix_([1, 5, 7, 2], [0, 3, 1, 2])] # ix_는 index의 의미, 좀 더 직관적으로 이해하기 쉬움.


#이해를 위한 코드
ar = np.arange(8)
ar
ar2 = ar.reshape(2, 4)
ar2
ar2.shape
ar2[1]
ar2[:, 1]

ar3 = np.arange(60).reshape(3, 5, 4)
ar3

ar3[[1, 2], [4, 2], [0, 1]] # 1, 4, 2 / 2, 2, 1을 출력
ar3[np.ix_([1, 2], [4, 2], [0, 1])]  # 좌표값을 설정. 축은 1, 2 / 로우는 4, 2 / 컬럼은 0, 1


##배열 전치와 축 바꾸기
arr = np.arange(15).reshape((3, 5))
arr

arr.T # 축이 변경된 전치행렬 출력

#배열의 곱
arr = np.random.randn(6, 3)
np.dot(arr.T, arr)  # dot함수를 통해 배열의 곱셈가능. 당연히 m * n 행렬과 x * y일 때 n과 x의 값이 같아야 한다.


arr = np.arange(16).reshape((2, 2, 4))
arr
arr.transpose((1, 2, 0))  # T속성은 transpose함수와 같은 기능. 그런데 transpose는 지정한 상태로 축 위치 변경
                          # 즉, 0번 축은 1번으로, 1번 축은 2번으로 등등.
                          
arr
arr.swapaxes(1, 2)


####유니버설 함수

arr = np.arange(10)
np.sqrt(arr)
np.exp(arr)


# 유니버설 함수가 아니라면
np.array([np.sqrt(x) for x in arr])  # 각각의 스칼라 값들을 계산해서 리스트에 넣고 ndarray로 변환함.    


x = np.random.randn(8)
y = np.random.randn(8)
x
y
np.maximum(x, y) # element-wise maximum 즉, 하나의 element마다 계산

arr = np.random.randn(7) * 5
np.modf(arr) # 정수 부분과 소수점 이하 부분을 각각의 배열로 반환


###배열을 사용한 데이터 처리

points = np.arange(-5, 5, 0.01) # 1000 equally spaced points
points

xs, ys = np.meshgrid(points, points)  # mesh는 그물, grid는 격자, 즉, -5 ~ 5까지 손쉽게 만들어줌
                                      # x와 y들의 좌표를 전부 그려주는 것.
ys
points.shape
type(xs)

from matplotlib.pyplot import imshow, title

import matplotlib.pyplot as plt  # 일반적으로 이렇게 정해져 있으므로 이에 따르자.
z = np.sqrt(xs ** 2 + ys ** 2)   # 10만개의 각각의 elements 들을 제곱하고 더해서 제곱근을 반환
z
plt.imshow(z, cmap=plt.cm.gray); plt.colorbar()
plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")
plt.draw()

#배열을 사용한 데이터 처리
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])


result = [(x if c else y)
          for x, y, c in zip(xarr, yarr, cond)] # zip에 의해 x, y, c는 예를 들어 1.1, 2.1, True가 됨. 각각을 튜플로 만들고 리스트로 변환

result

result = np.where(cond, xarr, yarr) # np.where함수는 삼항식의 벡터화된 버젼으로 condition, True exp, False exp를 인자로 받는다.
result

arr = np.random.randn(4, 4)
arr

np.where(arr > 0, 2, -2)
np.where(arr > 0, 2, arr) # set only positive values to 2


# Not to be executed

result = []
for i in range(n):
    if cond1[i] and cond2[i]:
        result.append(0)
    elif cond1[i]:
        result.append(1)
    elif cond2[i]:
        result.append(2)
    else:
        result.append(3)
        
        
# Not to be executed

np.where(cond1 & cond2, 0,
         np.where(cond1, 1,
                  np.where(cond2, 2, 3)))
                  
#참고! 0을 제외한 숫자는 true, 0은 false
# Not to be executed

#아래와 같이 코드를 짜면 반드시 1개만 true가 된다. &연산이기 때문. 즉, result는 0, 1, 2, 3 중 하나가 나오게 된다.
result = 0 * (cond1 & cond2) + 1 * (cond1 & ~cond2)
       + 2 * (~cond1 & cond2) + 3 * (~cond1 & ~cond2)
       
##수학메서드, 통계메서드
arr = np.random.randn(5, 4) # normally-distributed data
arr.mean()
np.mean(arr)
arr.sum()


arr

arr.mean(axis=1)  # axis, 즉 축에 대한 인자를 받음. 좌측의 코드는 1번 축을 기준으로 평균을 구하는 것. 즉, row의 평균을 구하게 됨.
                  # 결과로 한 차수 낮은 배열을 반환함.
arr.sum(0)        # axis가 0이므로 column의 합

arr.max(1)
arr.max(0)

arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
arr.cumsum(0)   #누적합을 구함. 0, 0+3, 0+3+6..
arr.cumprod(1)  #누적곱

arr = np.random.randint(0 ,10, (3, 3)) #0 ~10 까지 3 * 3 배열로, 정수값만 반환, 난수
arr

arr.argmax(0)  #argument는 index를 의미함.
arr.argmax(1)


##불리언 배열을 위한 메서드
arr = np.random.randn(100)
(arr > 0)
(arr > 0).sum() # Number of positive values, 즉 true의 개수 리턴함.

bools = np.array([False, False, True, False])
bools.any() #하나라도 true이면됨.
bools.all() # 전부 true여야함

a = np.random.randint(0, 10, size=10)
a

a.any()
a.all()