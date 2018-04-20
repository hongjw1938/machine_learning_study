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



##함수 적용과 매핑
frame = DataFrame(np.random.randn(4, 3), columns=list('bde'),
                  index=['Utah', 'Ohio', 'Texas', 'Oregon'])
                  
frame
np.abs(frame) # 각 배열의 원소에 절대치가 적용되었음. 내부적으로 반복문이 수행되기 때문

#람다함수를 만들고 변수에 넣었음. 이러면 이름이 있는 함수.
f = lambda x: x.max() - x.min()
test = np.array([1, 2, 3, 4])
f(test) #이와 같이 사용할 수 있다.


#pandas의 apply함수 적용 이는 dataframe의 apply함수임.
#row 혹은 column으로 쪼개서 apply메서드에 의해 내부에 인자로 들어온 함수를 적용시키게 된다.
frame.apply(f)

#1번 축 방향으로 함수 적용함.
frame.apply(f, axis=1)

#이 경우는 Series형태로 리턴함. 0번축 기준으로 각 column에 대해 Series로 index를 주기 때문에 dataframe으로 리턴됨.
def f(x):
    return Series([x.min(), x.max()], index=['min', 'max'])
frame.apply(f)

format = lambda x: '%.2f' % x

#실수값을 문자열 포맷으로 변경
frame.applymap(format)


#실수값을 문자열 포맷으로 변경
format = lambda x: '%.2f' % x
frame.applymap(format)
'%.2f, %s' % (3.141592, 'Hello')

#map함수

#python의 map 함수 리뷰
l = [1, 3, 4, 2]
def f1(x):
    return x ** 2 - 3*x
    
#l ** 2  # R은 recycling 규칙이 있어서 가능하지만 파이썬은 그러한 규칙이 없어서 불가능하다. 
#그래서 위의 f1함수도 f1(l)처럼 사용할 수가 없다.
#따라서 for loop를 이용한 방법을 사용할 수 있다. (해결방법 1)
result = []
for x in l:
    result.append(x ** 2 - 3*x)
    
result

#혹은 map함수를 이용할 수 있다. (해결방법2)
result = list(map(f1, l))
result


#Series의 map 메서드
type(frame['e'])
#e컬럼 즉, Series에 map함수로 함수 인자를 주면 Series의 각 원소에 함수가 적용되어 반환된다. 
frame['e'].map(format)

##정렬과 순위

obj = Series(range(4), index=['d', 'a', 'b', 'c'])
obj
#원본 데이터는 변경되지 않음.
obj.sort_index()
obj

frame = DataFrame(np.arange(8).reshape((2, 4)), index=['three', 'one'],
                  columns=['d', 'a', 'b', 'c'])
frame.sort_index()

frame.sort_index(axis=1) # 1번축 기준 알파벳 정렬

frame.sort_index(axis=1, ascending=False) #내림차순 정렬 가능

#sort_index는 deprecate warning이 있으므로 sort_values사용
#frame.sort_index(by='b')
#frame.sort_values() 오류남
frame.sort_values(by='b')

#frame.sort_index(by=['a', 'b'])
#a 컬럼 우선 정렬
frame.sort_values(by=['a', 'b'])


#순위
obj = Series([4, 7, -3, 2])
obj.rank()

obj = Series([7, -5, 7, 4, 2, 0, 4])
obj.rank() #같은 값이 여러개 있으면 등수를 평균하여 계산.

obj.rank(method='first') #rank를 정하는데 같은 값이 나오면 먼저 나온 value가 등수가 높다.

obj
obj.rank(ascending=False, method='max') # 큰 값이 등수가 높으며 같은 값의 경우 등수가 같음. 이 때 갯수만큼 등수가 내려감.

frame.rank(axis=0) #등수 평균을 냈기 때문에 1.5, 3.5등이 나옴.
frame.rank(axis=1)

##중복색인

obj = Series(range(5), index=['a', 'a', 'b', 'b', 'c'])
obj


obj.index.is_unique #색인이 유니크 한지 확인하는 bool 반환
obj['a'] #Series에서 값이 2개가 나왔으니 반환도 Series
obj['c']

df = DataFrame(np.random.randn(4, 3), index=['a', 'a', 'b', 'b'])
df
df.loc['b']
df.iloc[2:]


##기술통계 계산과 요약
df = DataFrame([[1.4, np.nan], [7.1, -4.5],
                [np.nan, np.nan], [0.75, -1.3]],
               index=['a', 'b', 'c', 'd'],
               columns=['one', 'two'])
df

df.sum() #축이 생략되어서 axis=0가 default, NaN값은 무시하고 계산된다.
df.sum(axis=1)

df.mean(axis=1, skipna=False) #누락 데이터를 제외하지 않고 연산한다.

df.idxmax() #최대값을 가지는 색인 반환
df.max()
df.loc[['b', 'd']]

df.cumsum() #누적 합.
df.describe() #갯수, 평균, 표준편차, 최소값, 분위수, 최대값을 보여줌

obj = Series(['a', 'a', 'b', 'c'] * 4)
obj.describe() #데이터가 명목척도와 같으면 평균등을 구할 수 없다. 따라서 빈도수, unique값, 최빈 값등을 연산해서 리턴한다.


##유일 값, 값 세기, 멤버쉽
obj = Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])

uniques = obj.unique()
uniques

obj.value_counts()

pd.value_counts(obj.values, sort=False)

mask = obj.isin(['b', 'c'])
mask

obj[mask]


data = DataFrame({'Qu1': [1, 3, 4, 3, 4],
                  'Qu2': [2, 3, 1, 2, 3],
                  'Qu3': [1, 5, 2, 4, 4]})
data

result = data.apply(pd.value_counts).fillna(0) #fillna는 비어있는 값을 채워주는 함수
result


####누락 데이터 처리

string_data = Series(['aardvark', 'artichoke', np.nan, 'avocado'])
string_data


string_data.isnull()


##누락 데이터 골라내기
from numpy import nan as NA
data = Series([1, NA, 3.5, NA, 7])
data.dropna() #실제 들어있는 색인 및 데이터를 Series로 반환


data
data[data.notnull()]

data = DataFrame([[1., 6.5, 3.], [1., NA, NA],
                  [NA, NA, NA], [NA, 6.5, 3.]])
cleaned = data.dropna() #data에서 NaN값을 drop함.
data

cleaned

data.dropna(how='all') #모든 값들이 NaN일 때만 로우를 드랍하고 하나라도 NaN이 아니면 남겨둔다.

data[4] = NA #컬럼4의 Series값들을 전부 NaN으로 변경
data

data.dropna(axis=1, how='all') #1번축에 대해서 전체 값이 NaN인 컬럼을 드랍

df = DataFrame(np.random.randn(7, 3))
df.iloc[:4, 1] = NA; df.iloc[:2, 2] = NA
df
df.dropna(thresh=3) #thresh는 기준값, NaN이 3개 이하이면 drop


##누락 값 채우기
df.fillna(0)


df.fillna({1: 0.5, 3: -1}) #column index를
# always returns a reference to the filled object
_ = df.fillna(0, inplace=True) #기존 객체를 변경함. 원래는 새로운 객체 반환 
df

df = DataFrame(np.random.randn(6, 3))
df.iloc[2:, 1] = NA; df.iloc[4:, 2] = NA
df

df.fillna(method='ffill') # 앞에 있는 값으로 채움. 0번축 기준.

df.fillna(method='ffill', limit=2) # limit옵션을 통해 최대 몇 개까지 채울 건지 지정 가능

data = Series([1., NA, 3.5, NA, 7])
data.fillna(data.mean()) #fillna를 하는데 계산을 통해 나온 평균 값으로 채워라는 의미

#퀴즈: df 데이터 프레임에서 각 컬럼의 NaN값을 해당 컬럼의 평균값으로 채우세요.
df.fillna({0: df.iloc[:,0].mean(), 1: df.iloc[:, 1].mean(), 2: df.iloc[:, 2].mean()})

#혹은
df.fillna(df.mean())


###계층적색인

data = Series(np.random.randn(10),
              index=[['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'd', 'd'],
                     [1, 2, 3, 1, 2, 3, 1, 2, 2, 3]])
data #multiindex를 색인으로 하는 Series임.

data.index
type(data.index) #Multiindex 객체임을 알 수 있다.

data['b'] # 상위 Index인 b의 하위 인덱스 및 값이 반환된다.
data['b':'c'] #end point도 반환됨

data.loc[['b', 'd']]

data[:, 2] #하위 계층 인덱스 선택 가능

data.unstack() #index를 옮겨줌. 하위 인덱스를 로우로 옮겨줌. default는 level=-1이다. 가장 하위 level을 의미함.data.unstack() #index를 옮겨줌. 하위 인덱스를 로우로 옮겨줌
data.unstack().stack() #stack은 로우에 있는 인덱스를 컬럼쪽으로 옮김

frame = DataFrame(np.arange(12).reshape((4, 3)),
                  index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                  columns=[['Ohio', 'Ohio', 'Colorado'],
                           ['Green', 'Red', 'Green']])
frame

#이름지정 가능
frame.index.names = ['key1', 'key2']
frame.columns.names = ['state', 'color']
frame

frame['Ohio']
MultiIndex.from_arrays([['Ohio', 'Ohio', 'Colorado'], ['Green', 'Red', 'Green']],
                       names=['state', 'color'])
                       
##계층순서 바꾸고 정렬하기
frame.swaplevel('key1', 'key2')
frame.sortlevel(1)
frame.swaplevel(0, 1).sortlevel(0)

frame.sortlevel(1) #예전 방식임. 단일 계층에 속한 데이터를 정렬함. 사전식으로 정렬함.
frame.swaplevel(0, 1).sortlevel(0)


##단계별 요약통계

frame.sum(level='key2') #key2에 따라 합을 구함.
frame.sum(level='color', axis=1) #color를 기준으로 어떤 축에 대해 더할지 지정.



##데이터프레임의 칼럼 사용하기
frame = DataFrame({'a': range(7), 'b': range(7, 0, -1),
                   'c': ['one', 'one', 'one', 'two', 'two', 'two', 'two'],
                   'd': [0, 1, 2, 0, 1, 2, 3]})
frame

frame2 = frame.set_index(['c', 'd']) #index를 지정하는 함수 즉, c, d컬럼의 값을 index로 지정함.
frame2

frame.set_index(['c', 'd'], drop=False)

frame2.reset_index() #index를 다시 컬럼으로 보냄.


###pandas와 관련된 기타 주제


## 정수 색인
ser = Series(np.arange(3.))
ser

ser.iloc[-1]
#ser[-1] #indexing할 때 정수값을 주면 레이블로 인식하기 때문에 찾지 못하게 된다.


ser2 = Series(np.arange(3.), index=['a', 'b', 'c'])
ser2
ser2[-1] #정수색인이 아니어서 문제가 없다.

ser2.iloc[:1]

ser3 = Series(range(3), index=[-5, 1, 3])
ser3.iloc[2] #3번째 데이터의 의미

frame = DataFrame(np.arange(6).reshape((3, 2)), index=[2, 0, 1])
frame

frame.iloc[0]
frame[0] #첫 번째 column을 반환


## panel 데이터