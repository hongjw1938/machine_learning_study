
# coding: utf-8

# ## Data Wrangling: Clean, Transform, Merge, Reshape

# In[1]:


from __future__ import division
from numpy.random import randn
import numpy as np
import os
import matplotlib.pyplot as plt
np.random.seed(12345)
plt.rc('figure', figsize=(10, 6))
from pandas import Series, DataFrame
import pandas
import pandas as pd
np.set_printoptions(precision=4, threshold=500)
pd.options.display.max_rows = 100


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'


# ## Combining and merging data sets

# ### Database-style DataFrame merges

# In[3]:


df1 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                 'data1': range(7)})
df2 = DataFrame({'key': ['a', 'b', 'd'],
                 'data2': range(3)})
df1


# In[4]:


df2


# In[7]:


pd.merge(df1, df2) #같은 이름인 컬럼을 찾아 inner join


# In[8]:


pd.merge(df1, df2, on='key')


# In[9]:


df3 = DataFrame({'lkey': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                 'data1': range(7)})
df4 = DataFrame({'rkey': ['a', 'b', 'd'],
                 'data2': range(3)})
df3
df4


# In[10]:


pd.merge(df3, df4, left_on='lkey', right_on='rkey')


# In[ ]:


df1
df2
pd.merge(df1, df2, how='outer') #외부조인


# In[ ]:


df1 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
                 'data1': range(6)})
df2 = DataFrame({'key': ['a', 'b', 'a', 'b', 'd'],
                 'data2': range(5)})
df1
df2


# In[ ]:


pd.merge(df1, df2, on='key', how='left')


# In[ ]:


pd.merge(df1, df2, how='inner')


# In[11]:


left = DataFrame({'key1': ['foo', 'foo', 'bar'],
                  'key2': ['one', 'two', 'one'],
                  'lval': [1, 2, 3]})
right = DataFrame({'key1': ['foo', 'foo', 'bar', 'bar'],
                   'key2': ['one', 'one', 'one', 'two'],
                   'rval': [4, 5, 6, 7]})
left
right
pd.merge(left, right, on=['key1', 'key2'], how='outer') #조인조건의 column이 2개 이상인 경우 리스트 형태로 지정


# In[12]:


pd.merge(left, right, on='key1')


# In[13]:


pd.merge(left, right, on='key1', suffixes=('_left', '_right')) #_x, _y가 아닌 다른 이름으로 붙이고 싶은 경우


# ### Merging on index

# In[14]:


left1 = DataFrame({'key': ['a', 'b', 'a', 'a', 'b', 'c'],
                  'value': range(6)})
right1 = DataFrame({'group_val': [3.5, 7]}, index=['a', 'b'])
left1
right1


# In[19]:


pd.merge(left1, right1, left_on='key', right_index=True)
pd.merge(right1, left1, right_on='key', left_index=True)


# In[16]:


pd.merge(left1, right1, left_on='key', right_index=True, how='outer')


# In[20]:


lefth = DataFrame({'key1': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
                   'key2': [2000, 2001, 2002, 2001, 2002],
                   'data': np.arange(5.)})
righth = DataFrame(np.arange(12).reshape((6, 2)),
                   index=[['Nevada', 'Nevada', 'Ohio', 'Ohio', 'Ohio', 'Ohio'],
                          [2001, 2000, 2000, 2000, 2001, 2002]],
                   columns=['event1', 'event2'])
lefth
righth


# In[21]:


pd.merge(lefth, righth, left_on=['key1', 'key2'], right_index=True)


# In[22]:


pd.merge(lefth, righth, left_on=['key1', 'key2'],
         right_index=True, how='outer')


# In[23]:


left2 = DataFrame([[1., 2.], [3., 4.], [5., 6.]], index=['a', 'c', 'e'],
                 columns=['Ohio', 'Nevada'])
right2 = DataFrame([[7., 8.], [9., 10.], [11., 12.], [13, 14]],
                   index=['b', 'c', 'd', 'e'], columns=['Missouri', 'Alabama'])
left2
right2


# In[24]:


pd.merge(left2, right2, how='outer', left_index=True, right_index=True) #양쪽 인덱스를 모두 merge key로 사용


# In[25]:


left2.join(right2, how='outer')


# In[26]:


left1
right1
left1.join(right1, on='key')


# In[27]:


left2
right2
another = DataFrame([[7., 8.], [9., 10.], [11., 12.], [16., 17.]],
                    index=['a', 'c', 'e', 'f'], columns=['New York', 'Oregon'])
another


# In[28]:


left2.join([right2, another])


# In[29]:


left2.join([right2, another], how='outer')


# In[30]:


left2
right2
another = DataFrame([[7., 8.], [9., 10.], [11., 12.], [16., 17.]],
                    index=['a', 'c', 'e', 'f'], columns=['New York', 'Ohio'])
another


# In[31]:


left2.join([right2, another]) #같은 이름의 컬럼이 있게 되면 join은 에러를 발생시킨다.


# ### Concatenating along an axis

# In[32]:


arr = np.arange(12).reshape((3, 4))
arr


# In[33]:


np.concatenate([arr, arr], axis=1) #1번 축을 기준으로 붙임. 컬럼을 이어 붙이는 것.
np.concatenate([arr, arr]) #0번 축 기준.


# In[34]:


s1 = Series([0, 1], index=['a', 'b'])
s2 = Series([2, 3, 4], index=['c', 'd', 'e'])
s3 = Series([5, 6], index=['f', 'g'])
s1
s2
s3


# In[35]:


pd.concat([s1, s2, s3])


# In[36]:


pd.concat([s1, s2, s3], axis=1)


# In[37]:


s4 = pd.concat([s1 * 5, s3])
s4


# In[38]:


pd.concat([s1, s4], axis=1)


# In[39]:


pd.concat([s1, s4], axis=1, join='inner') #inner조인으로 양 쪽에 있는 값만 출력


# In[40]:


s1
s4
pd.concat([s1, s4], axis=1, join_axes=[['a', 'c', 'b', 'e', 'wow']]) #index를 직접 지정 없으면 null값이 된다.


# In[41]:


result = pd.concat([s1, s1, s3], keys=['one', 'two', 'three']) #keys를 이용해 계층적 색인이 가능함.


# In[42]:


result


# In[43]:


# Much more on the unstack function later
result.unstack()


# In[44]:


pd.concat([s1, s2, s3], axis=1, keys=['one', 'two', 'three']) #axis가 1이라면 keys는 dataframe의 column이 된다.


# In[46]:


df1 = DataFrame(np.arange(6).reshape(3, 2), index=['a', 'b', 'c'],
                columns=['one', 'two'])
df2 = DataFrame(5 + np.arange(4).reshape(2, 2), index=['a', 'c'],
                columns=['three', 'four'])
df1
df2


# In[47]:


pd.concat([df1, df2], axis=1, keys=['level1', 'level2'])


# In[48]:


pd.concat({'level1': df1, 'level2': df2}, axis=1) #데이터를 사전 형태로 지정, keys를 주지 않아도 된다.


# In[49]:


pd.concat([df1, df2], axis=1, keys=['level1', 'level2'],
          names=['upper', 'lower']) #column의 이름을 지정할 수 있다.


# In[50]:


df1 = DataFrame(np.random.randn(3, 4), columns=['a', 'b', 'c', 'd'])
df2 = DataFrame(np.random.randn(2, 3), columns=['b', 'd', 'a'])
df1
df2


# In[51]:


pd.concat([df1, df2]) #컬럼이 같으므로 row로 붙이게 된다.


# In[52]:


pd.concat([df1, df2], ignore_index=True) #순서대로


# ### Combining data with overlap

# In[53]:


a = Series([np.nan, 2.5, np.nan, 3.5, 4.5, np.nan],
           index=['f', 'e', 'd', 'c', 'b', 'a'])
b = Series(np.arange(len(a), dtype=np.float64),
           index=['f', 'e', 'd', 'c', 'b', 'a'])
b[-1] = np.nan
a
b


# In[54]:


#pd.merge(a, b) #에러가 발생함
np.where(pd.isnull(a), b, a) #index가 서로 같으므로 where함수를 통해 어떤 것을 사용할지 지정한다.


# In[55]:


b[:-2].combine_first(a[2:]) #둘 다 null이면 null, 아니면 null이 아닌 것을 리턴


# In[56]:


df1 = DataFrame({'a': [1., np.nan, 5., np.nan],
                 'b': [np.nan, 2., np.nan, 6.],
                 'c': range(2, 18, 4)})
df2 = DataFrame({'a': [5., 4., np.nan, 3., 7.],
                 'b': [np.nan, 3., 4., 6., 8.]})
df1
df2
df1.combine_first(df2) #combine은 우선순위는 먼저인 값이며


# ## Reshaping and pivoting

# ### Reshaping with hierarchical indexing

# In[57]:


data = DataFrame(np.arange(6).reshape((2, 3)),
                 index=pd.Index(['Ohio', 'Colorado'], name='state'),
                 columns=pd.Index(['one', 'two', 'three'], name='number'))
data


# In[59]:


result = data.stack()
result
type(result)


# In[60]:


result.unstack()


# In[61]:


result.unstack(0)


# In[62]:


result.unstack('state')


# In[63]:


s1 = Series([0, 1, 2, 3], index=['a', 'b', 'c', 'd'])
s2 = Series([4, 5, 6], index=['c', 'd', 'e'])
s1
s2
data2 = pd.concat([s1, s2], keys=['one', 'two'])
data2


# In[64]:


data2.unstack()
data2.unstack(0)


# In[65]:


data2.unstack().stack()


# In[66]:


data2.unstack().stack(dropna=False) #unstack의 경우는 둘다 null이면 drop되지만, stack은 하나라도 null이면 drop
                                    #따라서 dropna=False라고 하면 null도 다 출력된다.


# In[67]:


result
df = DataFrame({'left': result, 'right': result + 5},
               columns=pd.Index(['left', 'right'], name='side'))
df


# In[68]:


df.unstack('state')


# In[70]:


test = df.unstack('state').stack('side')
test


# In[74]:


test1 = test.swaplevel('number', 'side').sort_index(0)
test1


# ### Pivoting "long" to "wide" format

# In[75]:


data = pd.read_csv('ch07/macrodata.csv')
data


# In[76]:


periods = pd.PeriodIndex(year=data.year, quarter=data.quarter, name='date')
periods


# In[77]:


data.to_records()  #튜플의 형태를 갖는 데이터 집합. 데이터 프레임으로 바꾸기 쉬운 형태로 변경


# In[81]:


data = DataFrame(data.to_records(),
                 columns=pd.Index(['realgdp', 'infl', 'unemp'], name='item'),
                 index=periods.to_timestamp('D', 'end')) #분기데이터를 날짜 데이터로 바꿈. 1959Q1이면 1959-03-31
data


# In[82]:


data.stack()


# In[83]:


data.stack().reset_index()


# In[79]:


data.stack().reset_index().rename(columns={0: 'value'})


# In[88]:


ldata = data.stack().reset_index().rename(columns={0: 'value'})
wdata = ldata.pivot('date', 'item', 'value') #row의 index가 1번째 인자, 2번째는 컬럼의 이름(item컬럼에 3가지 들어있음), 3번째는 데이터값


# In[86]:


ldata[:10]


# In[87]:


wdata


# In[89]:


pivoted = ldata.pivot('date', 'item', 'value')
pivoted.head()


# In[95]:


ldata['value2'] = np.random.randn(len(ldata))
ldata[:10]


# In[91]:


pivoted = ldata.pivot('date', 'item') #데이터 분석이 용이한 wide형으로 변형.
                                      #데이터값이 있는 컬럼명은 지정하지 않음. 전체 나머지 컬럼이 다 데이터값으로 됨.
pivoted[:5]


# In[92]:


pivoted['value'][:5]


# In[93]:


ldata.set_index(['date', 'item'])


# In[94]:


unstacked = ldata.set_index(['date', 'item']).unstack('item')
unstacked[:7]


# ## Data transformation

# ### Removing duplicates

# In[96]:


data = DataFrame({'k1': ['one'] * 3 + ['two'] * 4,
                  'k2': [1, 1, 2, 3, 3, 4, 4]})
data


# In[97]:


data.duplicated()


# In[98]:


data.drop_duplicates()


# In[100]:


data['v1'] = range(7)
data
data.drop_duplicates(['k1']) #k1을 고려해 중복인지 확인


# In[101]:


#data.drop_duplicates(['k1', 'k2'], take_last=True)
data.drop_duplicates(['k1', 'k2'], keep='last') #k1, k2만을 고려해서 중복인지 확인, last는 마지막으로 발견된 값 반환


# ### Transforming data using a function or mapping

# In[102]:


data = DataFrame({'food': ['bacon', 'pulled pork', 'bacon', 'Pastrami',
                           'corned beef', 'Bacon', 'pastrami', 'honey ham',
                           'nova lox'],
                  'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
data


# In[103]:


meat_to_animal = {
  'bacon': 'pig',
  'pulled pork': 'pig',
  'pastrami': 'cow',
  'corned beef': 'cow',
  'honey ham': 'pig',
  'nova lox': 'salmon'
}


# In[104]:


str.lower(data['food']) #문자열객체가 들어가야 하는데 Series객체를 넣었기 때문에 TypeError발생.


# In[105]:


get_ipython().run_line_magic('timeit', "data['food'].map(str.lower).map(meat_to_animal) #여러 번 수행해서 결과를 나타냄.")


# In[106]:


data['animal'] = data['food'].map(str.lower).map(meat_to_animal)
#map메서드는 사전류의 객체나 어떤 함수를 받을 수 있다.!
data


# In[107]:


get_ipython().run_line_magic('timeit', "data['food'].map(lambda x: meat_to_animal[x.lower()])")


# ### Replacing values

# In[108]:


data = Series([1., -999., 2., -999., -1000., 3.])
data


# In[110]:


data[data == -999] = np.nan
data


# In[111]:


data.replace(-999, np.nan)


# In[112]:


data.replace([-999, -1000], np.nan)


# In[113]:


data.replace([-999, -1000], [np.nan, 0])


# In[114]:


data.replace({-999: np.nan, -1000: 0})


# ### Renaming axis indexes

# In[115]:


data = DataFrame(np.arange(12).reshape((3, 4)),
                 index=['Ohio', 'Colorado', 'New York'],
                 columns=['one', 'two', 'three', 'four'])
data


# In[116]:


data.index
data.index.map(str.upper)


# In[117]:


data.index = data.index.map(str.upper)
data


# In[118]:


data.rename(index=str.title, columns=str.upper) #title형식, 대문자로 변경


# In[119]:


data.rename(index={'OHIO': 'INDIANA'},
            columns={'three': 'peekaboo'})


# In[120]:


data.rename(index={'OHIo': 'INDIANA', 'NEW YORK': 'NY'},
            columns={'three': 'peekaboo'})


# In[121]:


# Always returns a reference to a DataFrame
_ = data.rename(index={'OHIO': 'INDIANA'}, inplace=True)
data


# ### Discretization and binning

# In[122]:


ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]


# In[123]:


bins = [18, 25, 35, 60, 100]
cats = pd.cut(ages, bins) #default는 오른쪽이 닫힘구간.
cats #각 데이터에 대해 범위값을 반환.


# In[125]:


cats.codes #범위의 값들을 정수값으로 나타내준다.
#cats.labels


# In[126]:


cats.categories
#cats.levels


# In[127]:


pd.value_counts(cats) #구간별 도수를 반환


# In[128]:


pd.cut(ages, [18, 26, 36, 61, 100], right=False) #right를 False로 지정시 닫힘구간 열린구간을 반대로 바꿀 수 있다.


# In[129]:


group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
pd.cut(ages, bins, labels=group_names) #구간별 이름을 지정할 수 있다.


# In[130]:


pd.value_counts(pd.cut(ages, bins, labels=group_names))


# In[131]:


data = np.random.rand(20)
pd.value_counts(pd.cut(data, 4, precision=2)) #균등한 4개의 구간으로 나누고 유효숫자 2개를 나타낼 것.
data


# In[132]:


catal = pd.cut(data, 5)
data
catal


# In[133]:


pd.value_counts(catal)


# In[135]:


data = np.random.randn(1000) # Normally distributed
cats = pd.qcut(data, 4) # 구간에 포함되는 데이터의 갯수가 같도록 구간을 분할한다.
cats


# In[136]:


pd.value_counts(cats)


# In[137]:


pd.qcut(data, [0, 0.1, 0.5, 0.9, 1.]) #숫자를 넣게 되면 0.1은 10%, 0.5는 10~50%.. 식으로 구간을 형성한다.
pd.value_counts(pd.qcut(data, [0, 0.1, 0.5, 0.9, 1.]))


# ### Detecting and filtering outliers

# In[138]:


np.random.seed(12345)
data = DataFrame(np.random.randn(1000, 4))
data.head()
data.describe()


# In[139]:


col = data[3] #4번째 column을 col에 binding
col[np.abs(col) > 3] #ufunc을 사용해 절대치를 3과 비교.


# In[144]:


np.abs(data) > 3 #데이터 프레임 전체에 대해서 불리언 데이터프레임으로 반환시킴


# In[145]:


(np.abs(data) > 3).any(1)


# In[143]:


data[(np.abs(data) > 3).any(1)]


# In[147]:


np.abs(data) > 3


# In[148]:


data[np.abs(data) > 3] = np.sign(data) * 3
data.describe()


# In[149]:


data.head(10)


# ### Permutation and random sampling

# In[ ]:


df = DataFrame(np.arange(5 * 4).reshape((5, 4)))
sampler = np.random.permutation(5)
sampler


# In[ ]:


df


# In[ ]:


df.take(sampler)


# In[ ]:


df.take(np.random.permutation(len(df))[:3])


# In[ ]:


bag = np.array([5, 7, -1, 6, 4])
sampler = np.random.randint(0, len(bag), size=10)


# In[ ]:


sampler


# In[ ]:


draws = bag.take(sampler)
draws


# In[ ]:


bag[sampler]


# ## 로또 시뮬레이션

# In[ ]:


def show_me_the_lotto(money=10000, lotto_price=1000):
    return Series(np.arange(money/lotto_price))                .map(lambda x: np.sort(np.random.permutation(45)[:6] + 1))


# In[ ]:


show_me_the_lotto()


# ### Computing indicator / dummy variables

# In[ ]:


df = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
                'data1': range(6)})
df
pd.get_dummies(df['key'])


# In[ ]:


df[['data1']]


# In[ ]:


dummies = pd.get_dummies(df['key'], prefix='key')
dummies
df_with_dummy = df[['data1']].join(dummies)
df_with_dummy


# In[ ]:


mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table('ch02/movielens/movies.dat', sep='::', header=None,
                        names=mnames)
movies[:10]


# In[ ]:


genre_iter = (set(x.split('|')) for x in movies.genres)
genres = sorted(set.union(*genre_iter))
genres


# In[ ]:


dummies = DataFrame(np.zeros((len(movies), len(genres))), columns=genres)
dummies


# In[ ]:


gen1 = movies.genres[0]; gen1
gen1.split('|')


# In[ ]:


for i, gen in enumerate(movies.genres):
    dummies.loc[i, gen.split('|')] = 1


# In[ ]:


dummies


# In[ ]:


movies_windic = movies.join(dummies.add_prefix('Genre_'))
movies_windic.loc[0]


# In[ ]:


np.random.seed(12345)


# In[ ]:


values = np.random.rand(10)
values


# In[ ]:


bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
pd.get_dummies(pd.cut(values, bins))


# ## String manipulation

# ### String object methods

# In[ ]:


val = 'a,b,  guido'
val.split(',')


# In[ ]:


pieces = [x.strip() for x in val.split(',')]
pieces


# In[ ]:


first, second, third = pieces
first + '::' + second + '::' + third


# In[ ]:


'::'.join(pieces)


# In[ ]:


'guido' in val


# In[ ]:


val.index(',')


# In[ ]:


val.find(':')


# In[ ]:


val.index(':')


# In[ ]:


val.count(',')


# In[ ]:


val.replace(',', '::')


# In[ ]:


val.replace(',', '')


# ### Regular expressions

# In[ ]:


import re
text = "foo    bar\t baz  \tqux"
print(text)
text.split(' ')
re.split('\s+', text)
re.findall('\w+', text)


# In[ ]:


regex = re.compile('\s+')
regex.split(text)


# In[ ]:


regex.findall(text)


# In[ ]:


text = """Dave dave@google.com
Iceman iceman@snu.ac.kr
Steve steve@gmail.com
Rob rob@gmail.com
Ryan ryan@yahoo.com
"""
pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'

# re.IGNORECASE makes the regex case-insensitive
regex = re.compile(pattern, flags=re.IGNORECASE)


# In[ ]:


regex.findall(text)


# In[ ]:


m = regex.search(text)
m
type(m)


# In[ ]:


m.start()
m.end()
text[m.start():m.end()]


# In[ ]:


print(regex.match(text))


# In[ ]:


print(text)
print(regex.sub('REDACTED', text))


# In[ ]:


pattern = r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'
regex = re.compile(pattern, flags=re.IGNORECASE)


# In[ ]:


m = regex.match('wesm@bright.abc.net.com')
m.groups()
m.group(1)
m.group(0)


# In[ ]:


regex.findall(text)


# In[ ]:


print(text)


# In[ ]:


print(regex.sub(r'Username: \1, Domain: \2, Suffix: \3', text))


# In[ ]:


regex = re.compile(r"""
    (?P<username>[A-Z0-9._%+-]+)
    @
    (?P<domain>[A-Z0-9.-]+)
    \.
    (?P<suffix>[A-Z]{2,4})""", flags=re.IGNORECASE|re.VERBOSE)


# In[ ]:


m = regex.match('wesm@bright.net')
m.groupdict()


# In[ ]:


m_dict = m.groupdict()
m_dict['username']


# ### Vectorized string functions in pandas

# In[ ]:


data = {'Dave': 'dave@google.com', 'Steve': 'steve@gmail.com',
        'Rob': 'rob@gmail.com', 'Wes': np.nan}
data = Series(data)
data


# In[ ]:


data.isnull()
data[data.notnull()]


# In[ ]:


data.str.contains('gmail')


# In[ ]:


pattern


# In[ ]:


regex.findall(data)


# In[ ]:


data[data.notnull()].map(regex.findall)


# In[ ]:


df = DataFrame(data)
df


# In[ ]:


df[:-1].applymap(regex.findall)


# In[ ]:


data.str.findall(pattern, flags=re.IGNORECASE)


# In[ ]:


matches = data.str.match(pattern, flags=re.IGNORECASE)
matches


# In[ ]:


matches.str.get(1)


# In[ ]:


matches.str[0]


# In[ ]:


data.str[:5]


# ## Example: USDA Food Database
{
  "id": 21441,
  "description": "KENTUCKY FRIED CHICKEN, Fried Chicken, EXTRA CRISPY,
Wing, meat and skin with breading",
  "tags": ["KFC"],
  "manufacturer": "Kentucky Fried Chicken",
  "group": "Fast Foods",
  "portions": [
    {
      "amount": 1,
      "unit": "wing, with skin",
      "grams": 68.0
    },

    ...
  ],
  "nutrients": [
    {
      "value": 20.8,
      "units": "g",
      "description": "Protein",
      "group": "Composition"
    },

    ...
  ]
}
# In[ ]:


import json
db = json.load(open('ch07/foods-2011-10-03.json'))
len(db)
type(db)


# In[ ]:


db[0]


# In[ ]:


db[0].keys()


# In[ ]:


len(db[0]['nutrients'])
db[0]['nutrients'][0]


# In[ ]:


nutrients = DataFrame(db[0]['nutrients'])
nutrients[:7]


# In[ ]:


info_keys = ['description', 'group', 'id', 'manufacturer']
info = DataFrame(db, columns=info_keys)


# In[ ]:


info[:5]


# In[ ]:


info


# In[ ]:


pd.value_counts(info.group)[:10]


# In[ ]:


nutrients = []

for rec in db:
    fnuts = DataFrame(rec['nutrients'])
    fnuts['id'] = rec['id']
    nutrients.append(fnuts)

nutrients = pd.concat(nutrients, ignore_index=True)


# In[ ]:


nutrients


# In[ ]:


nutrients.duplicated().sum()


# In[ ]:


nutrients = nutrients.drop_duplicates()


# In[ ]:


col_mapping = {'description' : 'food',
               'group'       : 'fgroup'}
info = info.rename(columns=col_mapping, copy=False)
info


# In[ ]:


col_mapping = {'description' : 'nutrient',
               'group' : 'nutgroup'}
nutrients = nutrients.rename(columns=col_mapping, copy=False)
nutrients


# In[ ]:


ndata = pd.merge(nutrients, info, on='id', how='outer')


# In[ ]:


ndata


# In[ ]:


len(ndata)
ndata.loc[30000]


# In[ ]:


len(ndata[ndata.nutrient == 'Glycine'])


# In[ ]:


result = ndata.groupby(['nutrient', 'fgroup'])['value'].quantile(0.5)
#result['Zinc, Zn'].order().plot(kind='barh')
result['Zinc, Zn'].sort_values().plot(kind='barh')


# In[ ]:


by_nutrient = ndata.groupby(['nutgroup', 'nutrient'])

get_maximum = lambda x: x.xs(x.value.idxmax())
get_minimum = lambda x: x.xs(x.value.idxmin())

max_foods = by_nutrient.apply(get_maximum)[['value', 'food']]

# make the food a little smaller
max_foods.food = max_foods.food.str[:50]


# In[ ]:


max_foods.loc['Amino Acids']['food']

