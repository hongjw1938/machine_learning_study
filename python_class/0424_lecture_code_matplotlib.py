
# coding: utf-8

# # Plotting and Visualization

# In[31]:


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
import matplotlib


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline #웹브라우저 안에 그림을 그림.')


# In[3]:


get_ipython().run_line_magic('pwd', '#작업경로')


# In[4]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'


# ## A brief matplotlib API primer

# In[5]:


import matplotlib.pyplot as plt


# ### Figures and Subplots

# In[8]:


fig = plt.figure()

ax1 = fig.add_subplot(2, 2, 1) #객체는 크기가 2 * 2이고 4개의 서브플롯 중에서 첫 번째를 선택하겠다는 의미
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)

#3개의 서브플롯을 가지는 빈 matplotlib Figure를 생성했다.


# In[10]:


from numpy.random import randn
plt.plot(randn(50).cumsum(), 'k--') # 'k--'는 마크 스타일. 점/선등의 그래프 스타일을 지정함.
                                    # 가장 최근에 만들어진 서브플롯에 그림을 그린다.


# In[12]:


_ = ax1.hist(randn(100), bins=20, color='k', alpha=0.3) #100개 숫자를 갖는 랜덤 배열을 20개 만듦. alpha는 투명도
ax2.scatter(np.arange(30), np.arange(30) + 3 * randn(30))


# In[7]:


fig #figure객체를 담고 있음.


# In[ ]:


plt.close('all')    # matlplotlib inline 에서는 close() 필요없음
                    # 그림을 anaconda prompt와 같은 툴에서 따로 켜서 그린 경우, 지울 때 사용


# In[20]:


fig, axes = plt.subplots(2, 3) #총 6개의 서브플롯을 생성, fig와 axes는 다른 객체임. fig는 전체, axes는 각각의 서브플롯
fig

axes # axes는 내부적으로 2 * 3, 2차원 배열형태임. numpy배열로 받는다.

#### Adjusting the spacing around subplots
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=None, hspace=None) #좌우상하의 어떤 값을 지정하는 것 none이면 default로 지정


# In[ ]:


InteractiveShell.ast_node_interactivity = 'last'    # 별도의 셀에서 수행되어야 함


# In[25]:


fig, axes = plt.subplots(2, 2, sharex=True, sharey=True) #x축, y축을 같이 쓴다는 의미
for i in range(2):
    for j in range(2):
        axes[i, j].hist(randn(500), bins=50, color='k', alpha=0.5) #각각의 서브플랏에 그림을 그림.
plt.subplots_adjust(wspace=0, hspace=0) #폭, 높이의 간격을 0으로 지정


# ### Colors, markers, and line styles

# In[26]:


plt.figure()
plt.plot(randn(30).cumsum(), 'ko--')


# In[27]:


data = randn(30).cumsum()
plt.plot(data, 'k--', label='Default')
plt.plot(data, 'r-', drawstyle='steps-post', label='steps-post')
plt.legend(loc='best') #범례


# In[36]:


#ggplot style
ax3 = plt.gcf().add_subplot(212)
plt.plot(np.random.randn(30).cumsum(), 'red')


# In[42]:


#seaborn style
ax2 = fig.add_subplot(222)
plt.plot(np.random.randn(30).cumsum(), 'b.') #점으로 그림.


# In[43]:


ax3.clear()
plt.plot(np.random.randn(30).cumsum(), 'g--')


# ### Ticks, labels, and legends

# #### Setting the title, axis labels, ticks, and ticklabels

# In[28]:


fig = plt.figure(); ax = fig.add_subplot(1, 1, 1)
ax.plot(randn(1000).cumsum())

ticks = ax.set_xticks([0, 250, 500, 750, 1000]) #set_xticks는 0, 250, 500, 750, 1000에 표식을 내는 것.(x축)
labels = ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'],
                            rotation=30, fontsize='small')
ax.set_title('My first matplotlib plot') #가운데 점이 0.5, 1임. default
ax.set_xlabel('Stages')


# #### Adding legends

# In[29]:


fig = plt.figure(); ax = fig.add_subplot(1, 1, 1)
ax.plot(randn(1000).cumsum(), 'k', label='one')
ax.plot(randn(1000).cumsum(), 'k--', label='two')
ax.plot(randn(1000).cumsum(), 'k.', label='three')

ax.legend(loc='best')


# In[32]:


get_ipython().run_line_magic('pinfo', 'matplotlib.style.use')


# ### Annotations and drawing on a subplot

# In[48]:


from datetime import datetime

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

data = pd.read_csv('ch08/spx.csv', index_col=0, parse_dates=True)
spx = data['SPX']

spx.plot(ax=ax, style='k-')


# In[49]:


crisis_data = [
    (datetime(2007, 10, 11), 'Peak of bull market'),
    (datetime(2008, 3, 12), 'Bear Stearns Fails'),
    (datetime(2008, 9, 15), 'Lehman Bankruptcy')
]


# In[51]:


for date, label in crisis_data:
    ax.annotate(label, xy=(date, spx.asof(date) + 50),
                xytext=(date, spx.asof(date) + 200),
                arrowprops=dict(facecolor='black'),
                horizontalalignment='left', verticalalignment='top')


# In[50]:


# Zoom in on 2007-2010
ax.set_xlim(['1/1/2007', '1/1/2011'])
ax.set_ylim([600, 1800])

ax.set_title('Important dates in 2008-2009 financial crisis')


# In[45]:


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

rect = plt.Rectangle((0.2, 0.75), 0.4, 0.15, color='k', alpha=0.3)
circ = plt.Circle((0.7, 0.2), 0.15, color='b', alpha=0.3)
pgon = plt.Polygon([[0.15, 0.15], [0.35, 0.4], [0.2, 0.6]],
                   color='g', alpha=0.5)

ax.add_patch(rect)
ax.add_patch(circ)
ax.add_patch(pgon)


# ### Saving plots to file

# In[46]:


fig


# In[ ]:


fig.savefig('figpath.svg')


# In[ ]:


fig.savefig('figpath.png', dpi=400, bbox_inches='tight')


# In[ ]:


from io import BytesIO
buffer = BytesIO()
plt.savefig(buffer)
plot_data = buffer.getvalue()


# ### matplotlib configuration

# In[47]:


#plt.rc('figure', figsize=(10, 10))
# default size: (6, 4)
plt.rc('figure', figsize=(6, 4)) #resource구성, matplotlibrc파일에 저장됨.


# ## Plotting functions in pandas

# ### Line plots

# In[52]:


plt.close('all')


# In[53]:


s = Series(np.random.randn(10).cumsum(), index=np.arange(0, 100, 10)) #index가 x축의 값
s.plot()


# In[54]:


df = DataFrame(np.random.randn(10, 4).cumsum(0),
               columns=['A', 'B', 'C', 'D'],
               index=np.arange(0, 100, 10)) #index가 x축의 값
df


# In[55]:


df.plot() #dataframe은 자동적으로 legend출력됨


# ### Bar plots

# In[56]:


fig, axes = plt.subplots(2, 1)
data = Series(np.random.rand(16), index=list('abcdefghijklmnop'))
data.plot(kind='bar', ax=axes[0], color='b', alpha=0.7)
data.plot(kind='barh', ax=axes[1], color='g', alpha=0.7)


# In[57]:


df = DataFrame(np.random.rand(6, 4),
               index=['one', 'two', 'three', 'four', 'five', 'six'],
               columns=pd.Index(['A', 'B', 'C', 'D'], name='Genus'))
df
df.plot(kind='bar') #Genus가 범례의 이름이 되었음


# In[ ]:


#plt.figure()


# In[58]:


df.plot(kind='barh', stacked=True, alpha=0.5) #쌓음


# In[59]:


InteractiveShell.ast_node_interactivity = 'all'


# In[60]:


#테이블당 나누어서 흡연/비흡연등으로 나누고 누가 팁을 많이 주는지.
tips = pd.read_csv('ch08/tips.csv')
#party_counts = pd.crosstab(tips.day, tips.size)  # tips.size 에서 오류
party_counts = pd.crosstab(tips.day, tips['size'])
party_counts
# Not many 1- and 6-person parties
#party_counts = party_counts.ix[:, 2:5]
party_counts = party_counts.loc[:, 2:5]    # 여기서 2:5는 index가 아니고 label임, 5도 포함됨
party_counts


# In[61]:


# Normalize to sum to 1
#party_pcts = party_counts.div(party_counts.sum(1).astype(float), axis=0)
party_pcts = party_counts.div(party_counts.sum(1), axis=0)
party_pcts

party_pcts.plot(kind='bar', stacked=True)


# ### Histograms and density plots

# In[ ]:


#plt.figure()


# In[62]:


tips['tip_pct'] = tips['tip'] / tips['total_bill'] #팁 비율에 대한 히스토그램을 그린다.
tips['tip_pct'].hist(bins=50)


# In[ ]:


#plt.figure()


# In[64]:


tips['tip_pct'].plot(kind='kde')


# In[ ]:


#plt.figure()


# In[65]:


comp1 = np.random.normal(0, 1, size=200)  # N(0, 1)
comp2 = np.random.normal(10, 2, size=200)  # N(10, 4)
values = Series(np.concatenate([comp1, comp2]))
values.hist(bins=100, alpha=0.3, color='k', normed=True)
values.plot(kind='kde', style='k--')


# ### Scatter plots

# In[67]:


macro = pd.read_csv('ch08/macrodata.csv')
macro.tail(5)    # 추가
data = macro[['cpi', 'm1', 'tbilrate', 'unemp']]
data.tail(5)    # 추가
trans_data = np.log(data).diff().dropna()
trans_data[-5:]
trans_data.tail(5)    # 추가


# In[ ]:


#plt.figure()


# In[68]:


plt.scatter(trans_data['m1'], trans_data['unemp'])
plt.title('Changes in log %s vs. log %s' % ('m1', 'unemp'))


# In[69]:


plt.rc('figure', figsize=(10, 10))
#pd.scatter_matrix(trans_data, diagonal='kde', color='k', alpha=0.3)
pd.plotting.scatter_matrix(trans_data, diagonal='kde', color='k', alpha=0.3)   # 변경
plt.rc('figure', figsize=(6, 4));


# ## Plotting Maps: Visualizing Haiti Earthquake Crisis data

# In[ ]:


data = pd.read_csv('ch08/Haiti.csv')
data[:5]
data.info()


# In[ ]:


data[['INCIDENT DATE', 'LATITUDE', 'LONGITUDE']][:10]


# In[ ]:


data['CATEGORY'][:6]


# In[ ]:


data.describe()


# In[ ]:


data = data[(data.LATITUDE > 18) & (data.LATITUDE < 20) &
            (data.LONGITUDE > -75) & (data.LONGITUDE < -70)
            & data.CATEGORY.notnull()]
data.head(5)
data.info()


# In[ ]:


def to_cat_list(catstr):
    stripped = (x.strip() for x in catstr.split(','))
    return [x for x in stripped if x]

def get_all_categories(cat_series):
    cat_sets = (set(to_cat_list(x)) for x in cat_series)
    return sorted(set.union(*cat_sets))

def get_english(cat):
    code, names = cat.split('.')
    if '|' in names:
        names = names.split(' | ')[1]
    return code, names.strip()


# In[ ]:


get_english('2. Urgences logistiques | Vital Lines')


# In[ ]:


data.CATEGORY.head(5)


# In[ ]:


all_cats = get_all_categories(data.CATEGORY)
all_cats[:5]
len(all_cats)
# Generator expression
english_mapping = dict(get_english(x) for x in all_cats)
english_mapping['2a']
english_mapping['6c']


# In[ ]:


def get_code(seq):
    return [x.split('.')[0] for x in seq if x]

all_codes = get_code(all_cats)
all_codes[:10]
code_index = pd.Index(np.unique(all_codes))
code_index
dummy_frame = DataFrame(np.zeros((len(data), len(code_index))),
                        index=data.index, columns=code_index)
dummy_frame.info()


# In[ ]:


len(all_codes)
len(np.unique(all_codes))


# In[ ]:


dummy_frame.head()


# In[ ]:


#dummy_frame.ix[:, :6].info()
dummy_frame.iloc[:, :6].info()


# In[ ]:


data.index[:5]
data.CATEGORY.head()


# In[ ]:


for row, cat in zip(data.index, data.CATEGORY):
    codes = get_code(to_cat_list(cat))
    #dummy_frame.ix[row, codes] = 1
    dummy_frame.loc[row, codes] = 1
dummy_frame.head()


# In[ ]:


data = data.join(dummy_frame.add_prefix('category_'))
data.head(5)


# In[ ]:


#data.ix[:, 10:15].info()
data.iloc[:, 10:15].info()


# In[ ]:


from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

def basic_haiti_map(ax=None, lllat=17.25, urlat=20.25,
                    lllon=-75, urlon=-71):
    # create polar stereographic Basemap instance.
    m = Basemap(ax=ax, projection='stere',
                lon_0=(urlon + lllon) / 2,
                lat_0=(urlat + lllat) / 2,
                llcrnrlat=lllat, urcrnrlat=urlat,
                llcrnrlon=lllon, urcrnrlon=urlon,
                resolution='f')
    # draw coastlines, state and country boundaries, edge of map.
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    return m


# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
fig.subplots_adjust(hspace=0.05, wspace=0.05)

to_plot = ['2a', '1', '3c', '7a']

lllat=17.25; urlat=20.25; lllon=-75; urlon=-71

for code, ax in zip(to_plot, axes.flat):
    m = basic_haiti_map(ax, lllat=lllat, urlat=urlat,
                        lllon=lllon, urlon=urlon)

    cat_data = data[data['category_%s' % code] == 1]

    # compute map proj coordinates.
    x, y = m(cat_data.LONGITUDE.values, cat_data.LATITUDE.values)

    m.plot(x, y, 'r.', alpha=0.5)
    ax.set_title('%s: %s' % (code, english_mapping[code]))


# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
fig.subplots_adjust(hspace=0.05, wspace=0.05)

to_plot = ['2a', '1', '3c', '7a']

lllat=17.25; urlat=20.25; lllon=-75; urlon=-71

def make_plot():

    for i, code in enumerate(to_plot):
        cat_data = data[data['category_%s' % code] == 1]
        lons, lats = cat_data.LONGITUDE, cat_data.LATITUDE

        ax = axes.flat[i]
        m = basic_haiti_map(ax, lllat=lllat, urlat=urlat,
                            lllon=lllon, urlon=urlon)

        # compute map proj coordinates.
        x, y = m(lons.values, lats.values)

        m.plot(x, y, 'k.', alpha=0.5)
        ax.set_title('%s: %s' % (code, english_mapping[code]))
        
make_plot()        


# In[ ]:


plt.rc('figure', figsize=(12, 10))
fig = plt.figure()
cat_data = data[data['category_2a'] == 1]
lons, lats = cat_data.LONGITUDE, cat_data.LATITUDE

m = basic_haiti_map(lllat=33, urlat=39,
                    lllon=125, urlon=133)
#shapefile_path = 'ch08/PortAuPrince_Roads/PortAuPrince_Roads'
#m.readshapefile(shapefile_path, 'roads')
x, y = m(lons.values, lats.values)

m.plot(x, y, 'k.', alpha=0.5)
fig.get_axes()[0].set_title('Food shortages reported in Port-au-Prince')

