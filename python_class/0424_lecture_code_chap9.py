
# coding: utf-8

# # Data Aggregation and Group Operations

# In[1]:


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


# In[2]:


pd.options.display.notebook_repr_html = False


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'


# ## GroupBy mechanics

# In[5]:


df = DataFrame({'key1' : ['a', 'a', 'b', 'b', 'a'],
                'key2' : ['one', 'two', 'one', 'two', 'one'],
                'data1' : np.random.randint(10, size=5),
                'data2' : np.random.randint(10, size=5)})
df


# In[ ]:


grouped = df['data1'].groupby(df['key1']) #data1을 key1을 기준으로 그룹by
grouped


# In[7]:


for i in grouped:
    print(i)


# In[6]:


grouped2 = df.data1.groupby(df.key1) #위와 같은 방식
grouped2


# In[8]:


for i in grouped2:
    print(i)


# In[9]:


df['data1']
df.data1
df[['data1']].groupby(df.key1) #이 경우는 dataframe의 그룹by객체, 컬럼이 1개인 데이터 프레임이다.
                               #df['data1']은 Series이다.


# In[27]:


df[['data1']]


# In[19]:


for i in df[['data1']].groupby(df.key1):
    print(i)
test = df[['data1']].groupby(df.key1)
type(test)


# In[20]:


df['key1']
df.key1


# In[22]:


grouped
grouped.mean()


# In[23]:


[df.key1, df.key2]


# In[24]:


means = df['data1'].groupby([df['key1'], df['key2']]).mean()
means2 = df.data1.groupby([df.key1, df.key2]).mean()
means
means2


# In[25]:


means.unstack()


# In[26]:


df
states = np.array(['Ohio', 'California', 'California', 'Ohio', 'Ohio'])
years = np.array([2005, 2005, 2006, 2005, 2006])
df['data1'].groupby([states, years]).mean()


# In[28]:


df.groupby('key1').mean()


# In[29]:


df.groupby(['key1', 'key2']).mean()


# In[30]:


df.groupby(['key1', 'key2']).size()


# ### Iterating over groups

# In[31]:


for name, group in df.groupby('key1'): #key1에 따라 여러 그룹으로 나뉘게 됨.
    print(name)
    print(group)


# In[33]:


for (k1, k2), group in df.groupby(['key1', 'key2']): #각각의 key값을 k1,k2로 분리해서 받는다.
    print((k1, k2))
    print(group)
    print()


# In[32]:


for k, group in df.groupby(['key1', 'key2']):
    print(k)
    print(group)
    print()


# In[34]:


list(df.groupby('key1')) #튜플의 리스트로 반환됨.


# In[55]:


list(df.groupby('key1'))[1][1]['data1']
type(list(df.groupby('key1'))[1][1]) #dataframe객체임


# In[46]:


pieces = dict(list(df.groupby('key1')))
pieces
pieces['b']
pieceTest = pieces['b']


# In[43]:


pieceTest['data1']


# In[57]:


df.groupby('key1')
list(df.groupby('key1'))
dict(list(df.groupby('key1')))


# In[58]:


dict(df.groupby('key1'))


# In[59]:


df.dtypes
type(df.dtypes)


# In[60]:


df
grouped = df.groupby(df.dtypes, axis=1)
dict(list(grouped)) #dtype에 따라 그룹을 묶을 수도 있다.


# In[63]:


s1 = df.dtypes[[3, 0, 1, 2]]; s1


# In[62]:


grouped2 = df.groupby(s1, axis=1)
df
dict(list(grouped))


# ### Selecting a column or subset of columns

# In[64]:


df.groupby('key1')['data1']
df.groupby('key1')[['data2']]


# In[65]:


df['data1'].groupby(df['key1'])
df[['data2']].groupby(df['key1'])


# In[66]:


df.groupby(['key1', 'key2'])[['data2']].mean()


# In[67]:


s_grouped = df.groupby(['key1', 'key2'])['data2']
s_grouped


# In[68]:


s_grouped.mean()


# ### Grouping with dicts and Series

# In[69]:


people = DataFrame(np.random.randint(3, size=(5, 5)),
                   columns=['a', 'b', 'c', 'd', 'e'],
                  )
people.loc[2:3, ['b', 'c']] = np.nan # Add a few NA values
people


# In[70]:


people = DataFrame(np.random.randint(3, size=(5, 5)),
                   columns=['a', 'b', 'c', 'd', 'e'],
                   index=['Joe', 'Steve', 'Wes', 'Jim', 'Travis']
                  )
people.loc[2:3, ['b', 'c']] = np.nan # Add a few NA values
people


# In[71]:


mapping = {'a': 'red', 'b': 'red', 'c': 'blue',
           'd': 'blue', 'e': 'red', 'f' : 'orange'
          }
mapping


# In[72]:


by_column = people.groupby(mapping, axis=1)
list(by_column)
by_column.sum() #null값은 무시하고 연산함.


# In[73]:


map_series = Series(mapping)
map_series


# In[74]:


people.groupby(map_series, axis=1).count()
people.groupby(map_series, axis=1).sum()


# ### Grouping with functions

# In[75]:


people

people.groupby(len).sum()
people.T.groupby(len, axis=1).sum()


# In[76]:


key_list = ['one', 'one', 'one', 'two', 'two']
people.groupby([len, key_list]).min()


# In[77]:


people.groupby([len, 'd']).min()


# ### Grouping by index levels

# In[78]:


columns = pd.MultiIndex.from_arrays([['US', 'US', 'US', 'JP', 'JP'],
                                    [1, 3, 5, 1, 3]], names=['cty', 'tenor'])
columns
hier_df = DataFrame(np.random.randn(4, 5), columns=columns)
hier_df


# In[79]:


hier_df.groupby(level='cty', axis=1).count()


# ## Data aggregation

# In[80]:


df


# In[81]:


grouped = df.groupby('key1')
grouped['data1'].quantile(0.5)
grouped.quantile(0.9)['data1']


# In[82]:


def peak_to_peak(arr):
    return arr.max() - arr.min()
list(grouped)
grouped.agg(peak_to_peak)    # alias for aggregate()
grouped.aggregate(peak_to_peak)


# In[83]:


grouped.describe()


# In[84]:


tips = pd.read_csv('ch08/tips.csv')
# Add tip percentage of total bill
tips['tip_pct'] = tips['tip'] / tips['total_bill']
tips[:6]


# ### Column-wise and multiple function application

# In[85]:


grouped = tips.groupby(['sex', 'smoker']) #성별, 흡연여부로 그룹을 나눔
for (sex, smoker), group in grouped:
    print(sex, smoker)
    print(group)


# In[86]:


grouped_pct = grouped['tip_pct']
grouped['tip_pct'].mean() #각각의 그룹에 대해 평균을 구함
grouped_pct.agg('mean')


# In[87]:


grouped_pct.agg(['mean', 'std', peak_to_peak]) #함수이름을 문자열로 넘기면 된다.


# In[89]:


grouped_pct.agg([('foo', 'mean'), ('bar', np.std)])


# In[91]:


functions = ['count', 'mean', 'max']
result = grouped['tip_pct', 'total_bill'].agg(functions)
result


# In[92]:


result['tip_pct']


# In[93]:


ftuples = [('Durchschnitt', 'mean'), ('Abweichung', np.var)]
grouped['tip_pct', 'total_bill'].agg(ftuples)


# In[94]:


grouped.agg({'tip' : np.max, 'size' : 'sum'})


# In[95]:


grouped.agg({'tip_pct' : ['min', 'max', 'mean', 'std'],
             'size' : 'sum'})


# ### Returning aggregated data in "unindexed" form

# In[104]:


tips.groupby(['sex', 'smoker'], as_index=False).mean()


# ## 여성 비흡연자/흡연자, 남성 비흡연자/흡연자 별 평균 팁 비율을 막대그래프로, 여성/남성 흡연/비흡연자의 음식값과 팁 비율에 대한 산포도로 나타내 보세요

# In[97]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[98]:


#grouped_pct.mean().plot(kind='bar')
tips.groupby(['sex', 'smoker'])['tip_pct'].mean().plot(kind='barh')


# In[100]:


tip_f = tips[tips['sex'] == 'Female']
tip_fn = tip_f[tip_f['smoker'] == 'No']
tip_fy = tip_f[tip_f['smoker'] == 'Yes']
tip_m = tips[tips['sex'] == 'Male']
tip_mn = tip_m[tip_m['smoker'] == 'No']
tip_my = tip_m[tip_m['smoker'] == 'Yes']

plt.scatter(tip_fn['total_bill'], tip_fn['tip_pct'], label='Female No');
plt.scatter(tip_fy['total_bill'], tip_fy['tip_pct'], label='Female Yes', color='r');
plt.scatter(tip_mn['total_bill'], tip_mn['tip_pct'], label='Male No', color='b');
plt.scatter(tip_my['total_bill'], tip_my['tip_pct'], label='Male Yes', color='c');
plt.legend(loc='best');


# In[101]:


fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
axes = axes.flatten()

tip_f = tips[tips['sex'] == 'Female']
tip_fn = tip_f[tip_f['smoker'] == 'No']
tip_fy = tip_f[tip_f['smoker'] == 'Yes']
tip_m = tips[tips['sex'] == 'Male']
tip_mn = tip_m[tip_m['smoker'] == 'No']
tip_my = tip_m[tip_m['smoker'] == 'Yes']

axes[0].set_title('Female No-Smoker')
axes[0].scatter(tip_fn['total_bill'], tip_fn['tip_pct'], label='Female No');
axes[1].set_title('Female Smoker')
axes[1].scatter(tip_fy['total_bill'], tip_fy['tip_pct'], label='Female Yes', color='r');
axes[2].set_title('Male No-Smoker')
axes[2].scatter(tip_mn['total_bill'], tip_mn['tip_pct'], label='Male No', color='b');
axes[3].set_title('Male Smoker')
axes[3].scatter(tip_my['total_bill'], tip_my['tip_pct'], label='Male Yes', color='c');


# ## Group-wise operations and transformations

# In[105]:


df


# In[106]:


k1_means = df.groupby('key1').mean().add_prefix('mean_')
k1_means


# In[108]:


pd.merge(df, k1_means, left_on='key1', right_index=True)


# In[109]:


people


# In[110]:


key = ['one', 'two', 'one', 'two', 'one']
people.groupby(key).mean()


# In[111]:


list(people.groupby(key))
people.groupby(key).transform(np.mean)


# In[112]:


def demean(arr):
    return arr - arr.mean()
demeaned = people.groupby(key).transform(demean)
demeaned


# In[ ]:


demeaned.groupby(key).mean()


# ### Apply: General split-apply-combine

# In[114]:


tips[:10]
def top(df, n=5, column='tip_pct'):
    #return df.sort_index(by=column)[-n:]
    return df.sort_values(by=column)[-n:]
top(tips, n=6)


# In[115]:


tips.groupby('smoker').apply(top)


# In[116]:


tips.groupby(['smoker', 'day']).apply(top, n=2, column='total_bill') #최대에서 2개까지만 반환
tips.groupby(['smoker', 'day']).apply(top, n=1, column='total_bill')


# In[117]:


result = tips.groupby('smoker')['tip_pct'].describe()
result


# In[118]:


result.unstack('smoker')


# In[119]:


# tips.groupby('smoker')['tip_pct'].describe()는 내부적으로 아래와 같이 수행된다.
f = lambda x: x.describe()
tips.groupby('smoker')['tip_pct'].apply(f).unstack()


# #### Suppressing the group keys

# In[120]:


tips.groupby('smoker', group_keys=False).apply(top)


# ### Quantile and bucket analysis

# In[122]:


frame = DataFrame({'data1': np.random.randn(1000),
                   'data2': np.random.randn(1000)})
factor = pd.cut(frame.data1, 4) #data1을 4개의 구간으로 분할
frame.data1.describe()


# In[124]:


factor


# In[123]:


factor[:10]
type(factor)


# In[125]:


def get_stats(group):
    return {'min': group.min(), 'max': group.max(),
            'count': group.count(), 'mean': group.mean()}

grouped = frame.data2.groupby(factor)
grouped.groups


# In[133]:


grouped.apply(get_stats)
grouped.apply(get_stats).unstack()


#ADAPT the output is not sorted in the book while this is the case now (swap first two lines)


# In[140]:


# Return quantile numbers
grouping = pd.qcut(frame.data1, 10, labels=False)
grouping


# In[141]:


grouped = frame.data2.groupby(grouping) #data1을 qcut해 data2에 적용한 것.
grouped.apply(get_stats).unstack()


# ### Example: Filling missing values with group-specific values

# In[142]:


s = Series(np.random.randn(6))
s[::2] = np.nan
s
df = DataFrame(s)
df


# In[144]:


s.fillna(s.mean())
df.fillna(df.mean())


# In[145]:


['헬로']*3 + ['안녕']


# In[146]:


states = ['Ohio', 'New York', 'Vermont', 'Florida',
          'Oregon', 'Nevada', 'California', 'Idaho']
group_key = ['East'] * 4 + ['West'] * 4
data = Series(np.random.randn(8), index=states)
data[['Vermont', 'Nevada', 'Idaho']] = np.nan
data
group_key


# In[147]:


np.array(group_key) == 'East'


# In[148]:


data[np.array(group_key) == 'East']


# In[149]:


data.groupby(group_key).mean()


# In[150]:


fill_mean = lambda g: g.fillna(g.mean())
def fill_mean2(g):
    return g.fillna(g.mean)
list(data.groupby(group_key))
data.groupby(group_key).apply(fill_mean) 
data.groupby(group_key).transform(fill_mean)
data.groupby(group_key).apply(lambda g: g.fillna(g.mean()))


# In[151]:


fill_values = {'East': 0.5, 'West': -1}
fill_func = lambda g: g.fillna(fill_values[g.name])
def fill_func(g):
    return g.fillna(fill_values[g.name]) #g.name : 그룹의 이름을 반환

data.groupby(group_key).apply(fill_func)


# In[152]:


data.groupby(group_key)


# In[153]:


s1 = Series(np.arange(5)); s1
s2 = Series(np.arange(5), name='test'); s2
s1.name
s2.name


# In[154]:


Series(data.groupby(group_key).groups['East'], name='East').name


# ### Example: Random sampling and permutation

# In[156]:


# Hearts, Spades, Clubs, Diamonds
suits = ['H', 'S', 'C', 'D']
#card_val = (range(1, 11) + [10] * 3) * 4
card_val = (list(range(1, 11)) + [10]*3) * 4
card_val


# In[157]:


#base_names = ['A'] + range(2, 11) + ['J', 'K', 'Q']
base_names = ['A'] + list(range(2, 11)) + ['J', 'K', 'Q']
base_names


# In[159]:


cards = []
for suit in suits:
    cards.extend(str(num) + suit for num in base_names)
cards


# In[160]:


deck = Series(card_val, index=cards)


# In[161]:


card_val[:20]
base_names
deck[:13]


# In[162]:


def draw(deck, n=5):
    return deck.take(np.random.permutation(len(deck))[:n])
    #return deck[(np.random.permutation(len(deck))[:n])]  # fancy-indexing
draw(deck)


# In[166]:


deck[:]


# In[167]:


get_suit = lambda card: card[-1] # last letter is suit
deck.groupby(get_suit).apply(draw, n=2) #무늬별로 2장의 임의의 카드 뽑기


# In[164]:


# alternatively
deck.groupby(get_suit, group_keys=False).apply(draw, n=2)


# ### Example: Group weighted average and correlation

# In[ ]:


df = DataFrame({'category': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'],
                'data': np.random.randn(8),
                'weights': np.random.rand(8)})
df


# In[ ]:


grouped = df.groupby('category')
get_wavg = lambda g: np.average(g['data'], weights=g['weights'])
grouped.apply(get_wavg)


# In[ ]:


grouped.aggregate(get_wavg)


# In[ ]:


close_px = pd.read_csv('ch09/stock_px.csv', parse_dates=True, index_col=0)
close_px.info()


# In[ ]:


close_px[-4:]


# In[ ]:


dd = Series(np.random.randint(10, size=5)); dd
dd.pct_change()


# In[ ]:


rets = close_px.pct_change().dropna()
spx_corr = lambda x: x.corrwith(x['SPX'])
by_year = rets.groupby(lambda x: x.year)
by_year.apply(spx_corr)


# In[ ]:


# Annual correlation of Apple with Microsoft
by_year.apply(lambda g: g['AAPL'].corr(g['MSFT']))


# ### Example: Group-wise linear regression

# In[ ]:


import statsmodels.api as sm
from pandas.core import datetools
def regress(data, yvar, xvars):
    Y = data[yvar]
    X = data[xvars]
    X['intercept'] = 1.
    result = sm.OLS(Y, X).fit()
    return result.params


# In[ ]:


by_year.apply(regress, 'AAPL', ['SPX'])


# ## Pivot tables and Cross-tabulation

# In[ ]:


tips[:10]


# In[ ]:


tips.pivot_table(index=['sex', 'smoker'])
pd.pivot_table(tips, index=['sex', 'smoker'])


# In[ ]:


tips.pivot_table(['tip_pct', 'size'], index=['sex', 'day'],
                 columns='smoker')


# In[ ]:


tips.pivot_table(['tip_pct', 'size'], index=['sex', 'day'],
                 columns='smoker', margins=True, margins_name='Avg.')


# In[ ]:


tips.pivot_table('tip_pct', index=['sex', 'smoker'], columns='day',
                 aggfunc=len, margins=True)


# In[ ]:


tips.pivot_table('size', index=['time', 'sex', 'smoker'],
                 columns='day', aggfunc='sum', fill_value=0)


# ### Cross-tabulations: crosstab

# In[ ]:


#from StringIO import StringIO
from io import StringIO
data = """Sample    Gender    Handedness
1    Female    Right-handed
2    Male    Left-handed
3    Female    Right-handed
4    Male    Right-handed
5    Male    Left-handed
6    Male    Right-handed
7    Female    Right-handed
8    Female    Left-handed
9    Male    Right-handed
10    Female    Right-handed"""
data = pd.read_table(StringIO(data), sep='\s+')


# In[ ]:


data


# In[ ]:


pd.crosstab(data.Gender, data.Handedness, margins=True)


# In[ ]:


pd.crosstab([tips.time, tips.day], tips.smoker, margins=True)


# ## Example: 2012 Federal Election Commission Database

# In[ ]:


fec = pd.read_csv('ch09/P00000001-ALL.csv')


# In[ ]:


fec.info()


# In[ ]:


fec.loc[123456]


# In[ ]:


unique_cands = fec.cand_nm.unique()
unique_cands


# In[ ]:


unique_cands[2]


# In[ ]:


parties = {'Bachmann, Michelle': 'Republican',
           'Cain, Herman': 'Republican',
           'Gingrich, Newt': 'Republican',
           'Huntsman, Jon': 'Republican',
           'Johnson, Gary Earl': 'Republican',
           'McCotter, Thaddeus G': 'Republican',
           'Obama, Barack': 'Democrat',
           'Paul, Ron': 'Republican',
           'Pawlenty, Timothy': 'Republican',
           'Perry, Rick': 'Republican',
           "Roemer, Charles E. 'Buddy' III": 'Republican',
           'Romney, Mitt': 'Republican',
           'Santorum, Rick': 'Republican'}


# In[ ]:


fec.cand_nm[123456:123461]


# In[ ]:


fec.cand_nm[123456:123461].map(parties)


# In[ ]:


# Add it as a column
fec['party'] = fec.cand_nm.map(parties)


# In[ ]:


fec['party'].value_counts()


# In[ ]:


(fec.contb_receipt_amt > 0).value_counts()


# In[ ]:


fec = fec[fec.contb_receipt_amt > 0]
fec


# In[ ]:


fec_mrbo = fec[fec.cand_nm.isin(['Obama, Barack', 'Romney, Mitt'])]
fec_mrbo


# ### Donation statistics by occupation and employer

# In[ ]:


fec.contbr_occupation[:100]


# In[ ]:


fec.contbr_occupation.value_counts()[:10]


# In[ ]:


occ_mapping = {
   'INFORMATION REQUESTED PER BEST EFFORTS' : 'NOT PROVIDED',
   'INFORMATION REQUESTED' : 'NOT PROVIDED',
   'INFORMATION REQUESTED (BEST EFFORTS)' : 'NOT PROVIDED',
   'C.E.O.': 'CEO'
}
#occ_mapping['PROFESSOR']
o = 'PROFESSOR'
occ_mapping.get(o, o)


# In[ ]:


# If no mapping provided, return x
f = lambda x: occ_mapping.get(x, x)
fec.contbr_occupation = fec.contbr_occupation.map(f)
fec.contbr_occupation[:100]


# In[ ]:


emp_mapping = {
   'INFORMATION REQUESTED PER BEST EFFORTS' : 'NOT PROVIDED',
   'INFORMATION REQUESTED' : 'NOT PROVIDED',
   'SELF' : 'SELF-EMPLOYED',
   'SELF EMPLOYED' : 'SELF-EMPLOYED',
}

# If no mapping provided, return x
f = lambda x: emp_mapping.get(x, x)
fec.contbr_employer = fec.contbr_employer.map(f)


# In[ ]:


by_occupation = fec.pivot_table('contb_receipt_amt',
                                index='contbr_occupation',
                                columns='party', aggfunc='sum')
by_occupation[:10]
by_occupation.shape


# In[ ]:


over_2mm = by_occupation[by_occupation.sum(1) > 2000000]
over_2mm.shape
over_2mm


# In[ ]:


over_2mm.plot(kind='barh')


# In[ ]:


def get_top_amounts(group, key, n=5):
    totals = group.groupby(key)['contb_receipt_amt'].sum()

    # Order totals by key in descending order
    #return totals.sort(ascending=False)[-n:]
    return totals.sort_values(ascending=False)[:n]


# In[ ]:


grouped = fec_mrbo.groupby('cand_nm')
grouped.apply(get_top_amounts, 'contbr_occupation', n=7)


# In[ ]:


grouped.apply(get_top_amounts, 'contbr_employer', n=10)


# ### Bucketing donation amounts

# In[ ]:


bins = np.array([0, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000])
labels = pd.cut(fec_mrbo.contb_receipt_amt, bins)
labels


# In[ ]:


grouped = fec_mrbo.groupby(['cand_nm', labels])
grouped.size().unstack(0)


# In[ ]:


bucket_sums = grouped.contb_receipt_amt.sum().unstack(0)
bucket_sums


# In[ ]:


normed_sums = bucket_sums.div(bucket_sums.sum(axis=1), axis=0)
normed_sums


# In[ ]:


normed_sums[:-2].plot(kind='barh', stacked=True)


# ### Donation statistics by state

# In[ ]:


grouped = fec_mrbo.groupby(['cand_nm', 'contbr_st'])
totals = grouped.contb_receipt_amt.sum().unstack(0).fillna(0)
totals = totals[totals.sum(1) > 100000]
totals[:10]


# In[ ]:


percent = totals.div(totals.sum(1), axis=0)
percent[:10]

