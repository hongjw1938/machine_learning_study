##Pandas start

from pandas import Series, DataFrame
import pandas as pd

from __future__ import division
from numpy.random import randn
import numpy as np
import os
import matplotlib.pyplot as plt
np.random.seed(12345)
plt.rc('figure', figsize=(10, 6))
from pandas import Series, DataFrame
import pandas as pd
np.set_printoptions(precision=4)

%pwd #현재 작업 경로를 보여줌.

pwd #그냥 이렇게 써도 되는데 명시적으로 %쓰는 경우가 많음.


##pandas의 자료구조

#Series
#Series는 Class임, Numpy배열이나 어떤 자료구조도 가능. 심지어 Series객체도 가능
obj = Series([4, 7, -5, 3])
obj

obj.values #values속성, 값만 보여줌 numpy의 배열 반환하거나 ndarray의 객체와 유사한 것으로 반환
obj.index  #index속성, 

type(obj)
type(obj.values)
type(obj.index)

np.array(obj.index)

obj2 = Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c']) # index에 이름을 직접 주는 방식이 가능
obj2
obj2.index

obj2['a']

obj2['d'] = 6 # 값을 넣을 수 있음.
obj2[['c', 'a', 'd']] #지정한 순서대로 출력 가능

obj2 > 0 # bool값을 리턴


obj2[obj2 > 0]

obj2 * 2 #이 또한 Numpy의 브로드캐스팅의 규칙이 작용

np.exp(obj2) # pandas가 numpy기반이기 때문에 인자로 받아들임. 지수값을 반환

'b' in obj2 # 값이 아닌 인덱스를 기준으로 찾음.
'e' in obj2


sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = Series(sdata)
obj3 # 값이 2개의 컬럼처럼 보인다고 해도 Series는 1차원임. index는 값이 아님.

#obj3 시리즈에서 두 번째 원소의 값 출력
obj3[1]
obj3['Oregon']

# index의 이름을 리스트로 만들고 대입가능
states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = Series(sdata, index=states)
obj4 #California의 값은 dictionary에서 찾을 수 없기 때문에 NaN으로 출력


pd.isnull(obj4) # California index의 값은 True가 됨.
pd.notnull(obj4)

obj4.isnull() #위의 isnull()함수는 obj4의 인스턴스 메서드이기도 함. 따라서 좌측같은 코드도 가능

obj3 + obj4 # Index가 같은 것끼리 연산

#Series객체와 Series의 색인은 모두 name속성이 있음.
obj4.name = 'population'
obj4.index.name = 'state'
obj4

obj4.name
obj4.index.name

obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']
obj

#index는 unique하지 않아도 된다.

name = Series([1, 2, 3, 4], index=['Bob', 'Bob', 'Ryan', 'King'])
name + name

name2 = obj + name
name2

##Dataframe

data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
frame = DataFrame(data)
frame

DataFrame(data, columns=['year', 'state', 'pop']) #순서 변경하여 출력 가능



frame2 = DataFrame(data, columns=['year', 'state', 'pop', 'debt'],
                   index=['one', 'two', 'three', 'four', 'five'])
frame2 #debt 컬럼은 값이 없으므로 NaN

frame2.columns # index객체임. 즉, column도 index객체이다.

frame2['state'] #frame2는 year, state, pop, debt의 시리즈가 연결된 객체라고 볼 수 있다. 이 중, state만 뽑아냄

type(frame2['state']) # type을 출력해보면 Series객체임을 알 수 있다.


frame2.year #이와 같이 속성처럼 뽑을 수도 있음.


frame2.ix['three'] # index를 통해 각 Series객체의 값을 반환시킬 수 있음. 그러나, Deprecated되었으므로 사용하지 않을 것을 권함.
                   # loc 혹은 iloc를 사용할 것.
                   
frame2.loc['three']
#frame2.iloc['three'] # 반드시 정수값이 와야 하기 때문에 에러남.
frame2.iloc[2]

frame2['debt'] = 16.5 #특정 값을 넣을 수 있음
frame2

frame2['debt'] = np.arange(5.) #range로 값을 지정
frame2

val = Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
frame2['debt'] = val # two, four, five만 인덱스가 지정되어 있으므로 해당하는 값만 바꿈. 나머지는 값이 지정되어 있지 않으니 NaN
frame2

frame2['eastern'] = frame2.state == 'Ohio'
frame2 #eastern 컬럼을 만들고 bool Series객체를 삽입

del frame2['eastern'] #'eastern 삭제
frame2.columns

pop = {'Nevada': {2001: 2.4, 2002: 2.9}, #사전이 중첩됨. 
       'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
       
frame3 = DataFrame(pop)
frame3

frame3.T # T를 통해 Transpose 가능 
DataFrame(pop, index=[2001, 2002, 2003]) #색인 지정해서 dataframe생성 가능

pdata = {'Ohio': frame3['Ohio'][:-1], #마지막 row 직전까지, 만약 :로 해서 전체를 하면 Nevada는 두개만 출력하니까 마지막은 NaN이 됨.
         'Nevada': frame3['Nevada'][:2]} #0, 1 row만
DataFrame(pdata)

frame3.index.name = 'year'; frame3.columns.name = 'state' #index와 column의 이름 지정 가능
frame3

##색인객체
obj = Series(range(3), index=['a', 'b', 'c'])
index = obj.index
index

index[1:]

index[1] = 'd' #인덱스는 값을 변경할 수 없다. 따라서 자료구조 사이에서 안전하게 공유됨

index = pd.Index(np.arange(3))
index

obj2 = Series([1.5, -2.5, 0], index=index)
obj2
obj2.index is index #index객체인지 bool 값 반환

'Ohio' in frame3.columns

2002 in frame3.index
2003 in frame3.index

frame3.columns # column도 index로 다룬다.
'Ohio' in frame3.columns

###핵심기능

##재색인

#reindex
obj = Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
obj

obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
obj2

obj.reindex(['a', 'b', 'c', 'd', 'e'], fill_value=0) #빈 값은 0으로 채움

obj3 = Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
obj3.reindex(range(6), method='ffill') #앞의 값으로 채움

obj3
obj3.loc[2]
obj3.iloc[2] #이런 경우에 의해 인덱스를 정수로 주게 되면 혼란의 여지가 있다.
obj3[2] #이렇게 정수값으로 주게 되면 label이 되는 것.


obj2
obj2[2] #인덱스가 정수가 아니기 때문에 label로 인식하지 않는다.


frame = DataFrame(np.arange(9).reshape((3, 3)), index=['a', 'c', 'd'],
                  columns=['Ohio', 'Texas', 'California'])
frame

frame2 = frame.reindex(['a', 'b', 'c', 'd'])
frame2

states = ['Texas', 'Utah', 'California']
frame.reindex(columns=states)

#보간은 로우에 대해서만 가능, column, row를 동시 재색인 하면 보간도 양 쪽에 대해서 수행하려 하기 때문에 에러가 발생함.
frame.reindex(['a', 'b', 'c', 'd'], method='ffill', columns=states)

##로우 혹은 컬럼 삭제
#drop
obj = Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
new_obj = obj.drop('c')
new_obj

obj.drop(['d', 'c'])

data = DataFrame(np.arange(16).reshape((4, 4)),
                 index=['Ohio', 'Colorado', 'Utah', 'New York'],
                 columns=['one', 'two', 'three', 'four'])
                 
data.drop('two', axis=1) #axis를 주지 않으면 에러남
data.drop(['two', 'four'], axis=1)

##선택하기, 색인하기, 자르기
obj = Series(np.arange(4.), index=['a', 'b', 'c', 'd']) #정수로 색인하지 않아도 된다.
obj['b']
obj.b

obj[1]
obj.iloc[1] #label

obj[2:4]

obj[['b', 'a', 'd']]

obj[[1, 3]]

obj[obj < 2] #bool indexing

obj['b':'c'] #label로 인덱싱이 가능하다. 그런데 이 경우 end point의 값도 포함됨

obj['b':'c'] = 5
obj

data = DataFrame(np.arange(16).reshape((4, 4)),
                 index=['Ohio', 'Colorado', 'Utah', 'New York'],
                 columns=['one', 'two', 'three', 'four'])
data

data['two']
data[['three', 'one']]
data[:2]

data[data['three'] > 5]
data < 5 # bool값으로 채워진 행렬 반환

data[data < 5] = 0
data

data.loc['Colorado', ['two', 'three']] # Deprecation warning이 있으므로 loc사용
data.ix[['Colorado', 'Utah'], [3, 0, 1]] #이 경우는 loc를 사용시에 에러가 난다. 뒤의 정수도 label로 인식하기 때문이다.

#그래서 아래와 같이 교정
data.loc[['Colorado', 'Utah'], ['four', 'one', 'two']]
data.ix[2]

data.ix[:'Utah', 'two']
data.loc[:'Utah', 'two']

data.ix[data.three > 5, :3]
data.loc[data.three > 5, :'three']

##산술연산과 데이터 정렬
s1 = Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
s2 = Series([-2.1, 3.6, -1.5, 4, 3.1], index=['a', 'c', 'e', 'f', 'g'])

s1
s2

s1 + s2


df1 = DataFrame(np.arange(9.).reshape((3, 3)), columns=list('bcd'),
                index=['Ohio', 'Texas', 'Colorado'])
df2 = DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'),
                index=['Utah', 'Ohio', 'Texas', 'Oregon'])
df1
df2

df1 + df2

#산술연산 메서드에 채워 넣을 값 지정
df1 = DataFrame(np.arange(12.).reshape((3, 4)), columns=list('abcd')) # 문자열을 list에 넣었기에 분리됨
df2 = DataFrame(np.arange(20.).reshape((4, 5)), columns=list('abcde'))
df1

df2

df1 + df2

df1.add(df2, fill_value=0)

df1.reindex(columns=df2.columns, fill_value=0) #df1의 column을 df2의 column label로 채우고 빈 값은 0으로 지정

##데이터프레임과 Series간의 연산


arr = np.arange(12.).reshape((3, 4))
arr

arr[0]
arr - arr[0]


frame = DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'),
                  index=['Utah', 'Ohio', 'Texas', 'Oregon'])
series = frame.iloc[0]
frame
series

frame - series #Utah에 해당하는 column index를 찾아서 다른 row에도 같이 연산이 적용된다.


series2 = Series(range(3), index=['b', 'e', 'f'])
series2

frame
frame + series2 #d, f의 값이 지정되어 있지 않아서 NaN이 된다.



series3 = frame['d']
frame

series3

frame - series3 # 데이터 프레임은 열(0번축)으로 연산을 진행하는데 Utah, Ohio, Texas, Oregon이 없으니까 이에 대해 추가하고 나머지도 없으니 NaN이 된 것
frame.sub(series3, axis=0) #뺄셈을 적용하는데 적용하는 축 방향을 지정함.