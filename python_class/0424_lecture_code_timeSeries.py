
# coding: utf-8

# # Chapter 10 Time series

# In[1]:


from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
from numpy.random import randn
import numpy as np
pd.options.display.max_rows = 12
np.set_printoptions(precision=4, suppress=True)
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(12, 4))


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'


# ## 10. 1 Date and Time Data Types and Tools

# #### # 파이썬의 기본 날짜, 시간 자료형
#   * https://docs.python.org/3/library/datetime.html
#   
# #### # 날짜/시간 자료형
#   - datetime.date
#     + 그레고리언 달력 날짜(년, 월, 일)
#   - datetime.time
#     + 어느 하루의 시간을 시, 분, 초, 마이크로초 단위로 저장
#   - datetime.datetime
#     + 날짜와 시간을 같이 저장
#   - datetime.timedelta
#     + 두 datetime 값 간의 차이 (일, 초, 마이크로초)
#   - datetime.tzinfo
#   - datetime.timezone

# In[4]:


import datetime
datetime.MINYEAR
datetime.MAXYEAR
datetime


# In[ ]:


from datetime import datetime
datetime
datetime.MINYEAR
datetime.MAXYEAR


# - 위 에러 이유
# https://stackoverflow.com/questions/12906402/type-object-datetime-datetime-has-no-attribute-datetime

# In[ ]:


now = datetime.now()
now
now.year, now.month, now.day
now.hour, now.minute, now.second, now.microsecond


# In[ ]:


datetime.today()


# #### # timedelta 형

# In[ ]:


delta = datetime(2017, 9, 12) - datetime(2017, 8, 8, 8, 15)
delta
delta.days
delta.seconds
delta.microseconds


# In[ ]:


from datetime import timedelta
start = datetime(2011, 1, 7)
start + timedelta(12)


# In[ ]:


start - 2 * timedelta(12)


# #### # datetime ==> 문자열
#  * datetime.strftime('포맷 규칙')
# 
#  * format code
#    - https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
#   
# 
# * 교재 p.397~398

# In[ ]:


stamp = datetime(2011, 1, 3)


# In[ ]:


stamp
str(stamp)


# In[ ]:


stamp.strftime('%Y-%m-%d')
stamp.strftime('%F')
stamp.strftime('%a %A')


# #### # 문자열 ==> datetime
#   - datetime.strptime(datetime 객체, '포맷 규칙')

# In[ ]:


value = '2011-01-03'
datetime.strptime(value, '%Y-%m-%d')


# In[ ]:


datestrs = ['7/6/2011', '8/6/2011']
[datetime.strptime(x, '%m/%d/%Y') for x in datestrs]


# #### # dateutil 패키지는 거의 대부분의 사람이 인지하는 날짜 표현 방식 파싱 가능

# In[ ]:


from dateutil.parser import parse
parse('2011-01-03')


# In[ ]:


parse('Jan 31, 1997 10:45 PM')


# In[ ]:


parse('9/10-2017')


# In[ ]:


parse('9   10, 2017')


# #### # 유럽 여러 나라에서처럼 날짜가 월 앞에 오는 경우

# In[ ]:


parse('6/12/2011')
parse('6/12/2011', dayfirst=True)
parse('6/23/2011', dayfirst=True)
# parse('23/23/2011', dayfirst=True)  --> ValueError


# #### # pandas.to_datetime()
#  - 단일 날짜: pandas.Timestamp 형으로 변환
#  - 날짜의 배열: pandas.DatetimeIndex

# In[ ]:


pd.to_datetime('2017-9-17')


# In[ ]:


datestrs


# In[ ]:


pd.to_datetime(datestrs)


# #### # 누락된 날짜 데이터 처리
#   - NaT (Not a Time)

# In[ ]:


datestrs + [None]


# In[ ]:


idx = pd.to_datetime(datestrs + [None])
idx


# In[ ]:


idx[2]


# In[ ]:


pd.isnull(idx)


# ## 10.2 Time Series Basics

# #### # 시간 데이터
#   - Timestamp
#     : 시간 내 특정 순간
#   - Period
#     : 1년간, 1월간, 3주간, 분기간 등
#   - Interval
#     : 시작 타임스탬프, 끝 타임스탬프
#     : Period는 Interval의 특수한 경우
#   - Elapsed(Experiment) Time
#     : 특정 시작 시간에 대한 상대적인 시간의 측정 값
#   - Epoch
#     : Reference Time
#     - https://en.wikipedia.org/wiki/Epoch_(reference_date)
#     - https://stackoverflow.com/questions/1090869/why-is-1-1-1970-the-epoch-time

# #### # pandas 시간 데이터 타입
# |Class        |Remarks                       |How to create|
# |-------------|------------------------------|-------------|
# |Timestamp    |Represents a single time stamp|to_datetime, Timestamp|
# |DatetimeIndex|Index of Timestamp            |to_datetime, date_range, DatetimeIndex|
# |Period       |Represents a single time span |Period|
# |PeriodIndex  |Index of Period               |period_range, PeriodIndex|

# #### # 시계열 데이터
#   - 파이썬 문자열 또는 datetime 객체로 표현되는 Timestamp 인덱스의 Series 객체
#   - datetime 객체의 경우 Timestamp 타입으로 자동 변환되나 정렬되지 않음

# In[ ]:


from datetime import datetime
dates = [datetime(2011, 1, 2), datetime(2011, 1, 5), datetime(2011, 1, 7),
         datetime(2011, 1, 8), datetime(2011, 1, 10), datetime(2011, 1, 1)]

# index 인자로 입력된 datetime 리스트가 자동으로 DatetimeIndex 객체로 변환됨
ts1 = Series(np.random.randn(6), index=dates)
ts1


# In[ ]:


type(ts1)
# note: output changed to "pandas.core.series.Series"


# #### # DatetimeIndex의 스칼라 값: pandas의 Timestamp 객체

# In[ ]:


stamp = ts1.index[0]
stamp
# note: output changed from <Timestamp: 2011-01-02 00:00:00> to Timestamp('2011-01-02 00:00:00')
type(ts1.index)
type(ts1.index[0])


# In[ ]:


# index 인자로 날짜 문자열의 리스트를 대입하는 경우
da = ['2017-9-1', '2017-9-3', '2017-9-2', '2017-09-05', '2017-09-04']
ts2 = Series(np.random.randn(5), index=da); ts2
type(ts2.index)


# #### # 날짜 인덱스에 맞춰서 연산이 이루어짐

# In[ ]:


ts1.index
ts1


# In[ ]:


ts1[::2]
ts1 + ts1[::2]


# #### # Timestamp의 정밀도: 나노초(ns)

# In[ ]:


ts1.index.dtype
# note: output changed from dtype('datetime64[ns]') to dtype('<M8[ns]')


# ### 10.2.1 Indexing, selection, subsetting

# #### # pandas 시간 인덱스 하나의 값 접근하기(indexing)
#   - pandas Timestamp 객체
#   - python 문자열
#     - 시계열 데이터의 인덱스가 정렬되어 있지 않으면 Series 객체가 반환됨
#   - python datetime 객체

# In[ ]:


ts1
stamp = ts1.index[2]; stamp
ts1[stamp]

ts1['2011-01-07']
ts1['1-7/2011']
ts1['20110110']

ts1[datetime(2011, 1, 7)]


# In[ ]:


dates = [datetime(2011, 1, 2), datetime(2011, 1, 5), datetime(2011, 1, 7),
         datetime(2011, 1, 8), datetime(2011, 1, 10), datetime(2011, 1, 12)]
ts = Series(np.random.randn(6), index=dates)


# In[ ]:


ts
stamp = ts.index[2]; stamp
ts[stamp]

ts['2011-01-07']
ts['1-7/2011']
ts['20110110']

ts[datetime(2011, 1, 7)]


# #### # 년 또는 월만 넘겨서 해당 기간의 데이터만 선택

# In[ ]:


longer_ts = Series(np.random.randn(1000),
                   index=pd.date_range('8/1/2017', periods=1000))
longer_ts


# In[ ]:


longer_ts['2017']


# In[ ]:


longer_ts['2017-09']


# #### # 날짜로 데이터 자르기
#   - 원본 시계열 데이터에 대한 뷰가 됨

# In[ ]:


ts[datetime(2011, 1, 7):]
ts1[datetime(2011, 1, 7):]


# In[ ]:


ts
ts['1/6/2011':'1/11/2011']
ts['1/6/2011':'1/10/2011']


# #### 시계열 데이터를 특정 날짜를 기준으로 앞 또는 뒤 시계열 데이터를 버리기
#   - 특정 날짜는 버려지지 않음

# In[ ]:


ts
ts.truncate(after='1/8/2011')
ts.truncate(before='1/8/2011')
ts1.truncate(after='1/8/2011')
ts1.truncate(before='1/8/2011')


# #### # 시계열 데이터 접근(indexing) 방식은 데이터프레임에도 동일하게 적용
#   - 로우 인덱스에 적용

# In[ ]:


dates = pd.date_range('1/1/2017', periods=100, freq='W-WED')
dates


# In[ ]:


long_df = DataFrame(np.random.randn(100, 4),
                    index=dates,
                    columns=['Colorado', 'Texas', 'New York', 'Ohio'])
long_df
long_df.loc['9-2017']
long_df.loc['2017/09']


# ### 10.2.2 Time series with duplicate indices

# In[ ]:


dates = pd.DatetimeIndex(['1/1/2000', '1/2/2000', '1/2/2000', '1/2/2000',
                          '1/3/2000'])
dup_ts = Series(np.arange(5), index=dates)
dup_ts


# #### # 시계열 데이터의 인덱스가 유일한지 테스트

# In[ ]:


dup_ts.index.is_unique
ts.index.is_unique


# #### # 시계열 데이터 접근(indexing)
#   - 시간 인덱스가 유일 ==> 스칼라
#   - 시간 인덱스가 유일X ==> 시계열

# In[ ]:


dup_ts['1/3/2000']  # not duplicated


# In[ ]:


dup_ts['1/2/2000']  # duplicated


# #### # Timestamp 인덱스로 그룹 지어서 집계

# In[ ]:


grouped = dup_ts.groupby(level=0)
dup_ts
grouped.mean()


# In[ ]:


grouped.count()


# ## 10.3 Date ranges, Frequencies, and Shifting

# #### # 시간이 불규칙적인 시계열 ==> 고정 빈도 시계열
#   * resample API가 좀 더 groupby 처럼 변경됨 (v0.18.0)
#   * asfreq(): 빈도가 변경된 시계열을 반환
#     - https://pandas.pydata.org/pandas-docs/stable/generated/pandas.core.resample.Resampler.asfreq.html
#     
# * DateOffset objects
#   - https://pandas.pydata.org/pandas-docs/stable/timeseries.html#dateoffset-objects
# * Frequency strings (Offset aliases)
#   - https://pandas.pydata.org/pandas-docs/stable/timeseries.html#timeseries-offset-aliases

# In[ ]:


ts


# In[ ]:


ts.resample('D')


# In[ ]:


ts.resample('D').asfreq()


# In[ ]:


ts.resample('D').sum()


# In[ ]:


ts.resample('3d')
ts.resample('3d').sum()


# ### 10.3.1 Generating date ranges
#   * pandas.date_range(): DatetimeIndex 생성

# In[ ]:


index = pd.date_range('8/1/2017', '10/1/2017')
index


# In[ ]:


pd.date_range(start='8/1/2017', periods=20)


# In[ ]:


pd.date_range(end='10/1/2017', periods=20)


# In[ ]:


pd.date_range('2017-09-01', '2017-09-30', freq='B')


# #### # 매월 마지막 영업일 포함

# In[ ]:


pd.date_range('1/1/2017', '12/1/2017', freq='BM')
pd.date_range('1/1/2017', '12/31/2017', freq='BM')


# In[ ]:


pd.date_range('9/11/2017 12:56:31', periods=5)


# #### # 자정에 맞추어 타임스탬프 정규화(normalization)
#   - start, end 날짜를 자정으로 정규화한 후 date range 실행

# In[ ]:


pd.date_range('9/11/2017 12:56:31', periods=5, normalize=True)


# ### 10.3 Frequencies and Date Offsets
# * DateOffset objects
#   - https://pandas.pydata.org/pandas-docs/stable/timeseries.html#dateoffset-objects

# In[ ]:


from pandas.tseries.offsets import Hour, Minute
hour = Hour()
hour


# In[ ]:


four_hours = Hour(4)
four_hours
pd.Timestamp('2017-09-01') + four_hours


# #### # 위와 같이 객체를 직접 만들기 보단 아래처럼 간단한 문자열의 형태로 사용

# In[ ]:


pd.date_range('9/1/2017', '9/3/2017 23:59', freq=Hour(4))
pd.date_range('9/1/2017', '9/3/2017 23:59', freq='4h')


# In[ ]:


Hour(2) + Minute(30)


# In[ ]:


pd.date_range('9/1/2017', periods=10, freq='1h30min')


# #### # Anchored Offset
#   - 교재 표 10-4 p.408
#   - https://pandas.pydata.org/pandas-docs/stable/timeseries.html#anchored-offsets

# #### # 월별주차(WOM, Week of month)

# In[ ]:


# 매월 셋째 주 금요일
rng = pd.date_range('1/1/2017', '10/1/2017', freq='WOM-3FRI')
list(rng)
pd.date_range('1/1/2017', '10/1/2017', freq='3W-FRI')
pd.date_range('2017-09-13', periods=10, freq='3w')
pd.date_range(datetime.now(), periods=10, freq='3w')


# ### 10.3.3 Shifting (leading and lagging) data

# In[ ]:


ts = Series(np.random.randn(4),
            index=pd.date_range('9/1/2017', periods=4, freq='M'))
ts


# #### # 느슨한 시프트
#   - 시간 인덱스는 그대로, 데이터만 이동
#   - 데이터가 버려질 수 있음

# In[ ]:


ts.shift(2)


# In[ ]:


ts.shift(-2)


# #### # 시계열에서의 퍼센트 변화 계산 시

# In[ ]:


ts
ts.shift(1)
ts / ts.shift(1) - 1


# In[ ]:


ts.pct_change()


# #### # freq 인자에 같은 빈도를 주면 인덱스가 변경됨

# In[ ]:


ts.shift(2, freq='M')


# #### # freq 인자에 다른 빈도를 주면

# In[ ]:


ts
ts.shift(3, freq='D')


# In[ ]:


ts.shift(1, freq='3D')


# In[ ]:


ts.shift(1, freq='90T')


# #### # Shifting dates with offsets

# In[ ]:


from pandas.tseries.offsets import Day, MonthEnd
now = datetime(2011, 11, 17)
now + 3 * Day()


# In[ ]:


now + MonthEnd()


# In[ ]:


now + MonthEnd(2)


# #### # rollforward(), rollback()
#   - 해당 offset objects의 정수배만큼 입력된 날짜 데이터에 대해 적용

# In[ ]:


offset = MonthEnd()
offset.rollforward(now)    # now + MonthEnd()와 동일


# In[ ]:


offset.rollback(now)    # now - MonthEnd()와 동일


# In[ ]:


ts = Series(np.random.randn(20),
            index=pd.date_range('8/15/2017', periods=20, freq='4d'))
ts
ts.groupby(offset.rollforward).mean()


# In[ ]:


#ts.resample('M', how='mean')
ts.resample('M').mean()


# ## 10.4 Time Zone (시간대) Handling
#   * DST(일광절약시간, Day Saving Time, 서머 타임)
#   * UTC(국제표준시, Coordinated Universal Time, 협정세계시)

# #### # pytz 패키지
#   * Olson 시간대 데이터베이스를 기준으로 한, 역사적인 시간대와 현대적인 시간대를 모두 망라하고 있는 라이브러리
#     - https://pypi.python.org/pypi/pytz
#     - http://www.haruair.com/blog/1759

# In[ ]:


import pytz
pytz.common_timezones[-10:]
[ x for x in pytz.common_timezones if x.startswith('Asia')][:10]
[ x for x in pytz.common_timezones if x.endswith('Seoul')]


# #### # TimeZone 객체 생성하기

# In[ ]:


tz = pytz.timezone('Asia/Seoul')
tz


# ### 10.4.1 Localization and Conversion

# #### # 명시하지 않는한 시계열의 시간대(timezone)는 None

# In[ ]:


rng = pd.date_range('3/9/2012 9:30', periods=6, freq='D')
ts = Series(np.random.randn(len(rng)), index=rng)
ts


# In[ ]:


print(ts.index.tz)


# #### # 시간대 지정하여 날짜 범위 생성

# In[ ]:


pd.date_range('3/9/2012 9:30', periods=10, freq='D', tz='UTC')


# #### # 지정한 시간대로 시계열 변경
#   * 시간대가 없는 시계열: tz_localize()
#   * 시간대가 있는 시계열: tz_convert()
#     - 시간대가 다른 시계열로 변환

# In[ ]:


ts
ts_utc = ts.tz_localize('UTC')
ts_utc
ts


# In[ ]:


ts_utc.index
ts_utc.index.tzinfo


# In[ ]:


ts_utc.tz_convert('US/Eastern')


# In[ ]:


ts_eastern = ts.tz_localize('US/Eastern')
ts_eastern.index.tzinfo
ts_eastern.tz_convert('UTC')


# In[ ]:


ts_eastern.tz_convert('Europe/Berlin')


# In[ ]:


ts.index.tz_localize('Asia/Seoul')


# ### 10.4.2 Operations with time zone-aware Timestamp objects
#   * 시간대를 고려한 Timestamp 객체 다루기

# In[ ]:


stamp = pd.Timestamp('2011-03-12 04:00'); stamp
stamp_utc = stamp.tz_localize('utc'); stamp_utc
stamp_utc.tz_convert('US/Eastern')
stamp_utc.tz_convert('US/Pacific')
stamp_utc.tz_convert('Asia/Seoul')


# In[ ]:


show_time = pd.Timestamp('2017-09-11 10:00')
show_time_kr = show_time.tz_localize('Asia/Seoul'); show_time_kr
show_time_kr.tz_convert('US/Eastern')


# #### # Timezone 객체 생성 시 시간대 지정 가능

# In[ ]:


stamp_moscow = pd.Timestamp('2011-03-12 04:00', tz='Europe/Moscow')
stamp_moscow


# #### # UNIX epoch (1970년1월1일)부터 해당 시간까지의 나노초
#   - 시간대 변환해도 유지됨
#   - 즉, 일종의 절대 시간

# In[ ]:


stamp_utc.value


# In[ ]:


stamp_utc.tz_convert('US/Eastern').value


# #### # 일광절약시간제 고려한 시간 계산
#   * 일광절약시간제: https://ko.wikipedia.org/wiki/일광_절약_시간제
#   * DST 전환 시점 고려

# In[ ]:


# 30 minutes before DST transition
from pandas.tseries.offsets import Hour
stamp = pd.Timestamp('2012-03-11 01:30', tz='US/Eastern')
stamp


# In[ ]:


stamp + Hour()


# In[ ]:


# 90 minutes before DST transition
stamp = pd.Timestamp('2012-11-04 00:30', tz='US/Eastern')
stamp


# In[ ]:


stamp + 2 * Hour()


# ### 10.4.3 Operations between different time zones
#   * 서로 다른 시간대 연산 결과
#     - UTC

# In[ ]:


rng = pd.date_range('9/7/2017 9:30', periods=10, freq='B')
ts = Series(np.random.randn(len(rng)), index=rng)
ts


# In[ ]:


ts1 = ts[:7].tz_localize('Europe/London'); ts1.index
ts2 = ts1[2:].tz_convert('Europe/Moscow'); ts2.index
result = ts1 + ts2
result.index


# In[ ]:


ts1
ts2
result


# ## 10.5 Periods and Period Arithmetic
#   * 몇 일, 몇 개월, 몇 분기, 몇 해

# In[ ]:


p = pd.Period(2017, freq='A-DEC')
p


# In[ ]:


p + 5


# In[ ]:


p - 2


# In[ ]:


pd.Period('2014', freq='A-DEC') - p


# #### # pandas.period_range(): PeriodIndex 생성

# In[ ]:


rng = pd.period_range('1/1/2000', '6/30/2000', freq='M')
rng


# In[ ]:


Series(np.random.randn(6), index=rng)


# In[ ]:


values = ['2001Q3', '2002Q2', '2003Q1']
index = pd.PeriodIndex(values, freq='Q-DEC')
index


# ### 10.5.1 Period Frequency Conversion
#   * Period.asfreq()

# #### # 상위 단계 기간 ==> 하위 단계 기간

# In[ ]:


p = pd.Period('2007', freq='A-DEC');
p.asfreq('M', how='start')


# In[ ]:


p.asfreq('M', how='end')
p.asfreq('D', how='end')
p.asfreq('H', how='end')
p.asfreq('H', how='start')


# In[ ]:


p = pd.Period('2007', freq='A-JUN')
p.asfreq('M', 'start')


# In[ ]:


p.asfreq('M', 'end')


# #### # 하위 단계 기간 ==> 상위 단계 기간
#   - 상위 기간은 하위 기간이 어디에 속했는지에 따라 결정

# In[ ]:


p = pd.Period('Aug-2007', 'M'); p
p.asfreq('A-JUN')


# #### # PeriodIndex 객체도 마찬가지로 다루자

# In[ ]:


rng = pd.period_range('2006', '2009', freq='A-DEC'); rng
ts = Series(np.random.randn(len(rng)), index=rng)
ts


# In[ ]:


ts.asfreq
ts.asfreq('M', how='start')


# In[ ]:


ts.asfreq('B', how='end')


# ### 10.5.2 Quarterly period frequencies
#   * 회계 연도의 끝에 따라 의미가 달라짐
#   * 12 가지의 분기 빈도: Q-JAN ~ Q-DEC
#     - 4/4분기의 마지막 달이 Q- 다음에 오는 달

# In[ ]:


p = pd.Period('2017Q3', freq='Q-DEC')
p
p2 = pd.Period('2017Q3', freq='q-jan')
p2


# In[ ]:


p.asfreq('D', 'start')
p2.asfreq('D', 'S')


# In[ ]:


p.asfreq('D', 'end')
p2.asfreq('D', 'e')


# #### # 2017년 3분기 영업 마감일의 오후 4시

# In[ ]:


p4pm = (p.asfreq('B', 'e') - 1).asfreq('T', 's') + 16 * 60
p4pm


# In[ ]:


(p.asfreq('B', 'e').asfreq('T','s') + 16 * 60).to_timestamp()


# In[ ]:


p4pm.to_timestamp()


# #### # pandas.period_range()를 이용한 분기 범위 만들기

# In[ ]:


rng = pd.period_range('2016Q3', '2017Q4', freq='Q-JAN')
ts = Series(np.arange(len(rng)), index=rng)
ts


# In[ ]:


#new_rng = (rng.asfreq('B', 'e') - 1).asfreq('T', 's') + 16 * 60
new_rng = rng.asfreq('B', 'e').asfreq('T', 's') + 16 * 60
ts.index = new_rng.to_timestamp()
ts


# ### 10.5.3 Converting Timestamps to Periods (and back)
# * Timestamp <==> Period
# * to_period() <==> to_timestamp()

# In[ ]:


rng = pd.date_range('1/1/2000', periods=3, freq='M')
ts = Series(randn(3), index=rng)
ts


# #### # to_period()에 의해 변환되는 빈도(freq.)는 추정된다

# In[ ]:


pts = ts.to_period()
pts


# #### # 빈도를 지정할 수도 있음
#   * 중복되는 시간 인덱스가 나타날 수 있음

# In[ ]:


rng = pd.date_range('1/29/2000', periods=6, freq='D')
ts2 = Series(randn(6), index=rng)
ts2
ts2.to_period('M')


# In[ ]:


pts = ts.to_period(); pts


# In[ ]:


pts.to_timestamp(how='end')


# In[ ]:


ts2.to_period('M')
ts2.to_period('M').to_timestamp(how='end')


# ### 10.5.4 Creating a PeriodIndex from arrays

# In[ ]:


data = pd.read_csv('ch08/macrodata.csv')
data.year


# In[ ]:


data.quarter


# In[ ]:


index = pd.PeriodIndex(year=data.year, quarter=data.quarter, freq='Q-DEC')
index


# In[ ]:


data[:10]


# In[ ]:


data.index = index
data.infl
data[:10]


# ## 10.6 Resampling and Frequency Conversion
#   * Resampling: 시계열의 빈도를 변경하는 작업
#     - 다운샘플링: 상위 빈도 ==> 하위 빈도
#       - 표본을 천천히 뽑겠다
#       - 그룹 집계
#     - 업샘플링: 하위 빈도 ==> 상위 빈도
#       - 보간
#     - 사이드샘플링
#     
# 
#   * resample method
#     - Series.resample()
#     - DataFrame.resample()
#     - DataFrameGroupBy.resample()

# In[ ]:


rng = pd.date_range('1/1/2000', periods=100, freq='D')
ts = Series(randn(len(rng)), index=rng); ts
#ts.resample('M', how='mean')
ts.resample('M').mean()


# In[ ]:


#ts.resample('M', how='mean', kind='period')
ts.resample('M', kind='period').mean()


# ### Downsampling
#   * 샘플링(표본 추출) 비율을 낮추는 작업
#     - 표본을 천천히 뽑겠다
#   * 고려할 사항
#     - 각 간격의 양 끝 중에서 열어둘 쪽
#     - 집계하려는 구간의 레이블을 간격의 시작으로 할지 끝으로 할지 여부

# In[ ]:


rng = pd.date_range('1/1/2000', periods=12, freq='T')
ts = Series(np.arange(12), index=rng)
ts


# #### # 5분 단위로 묶고 묶은 결과는 합으로

# In[ ]:


#ts.resample('5min', how='sum')
ts.resample('5min').sum()
# note: output changed (as the default changed from closed='right', label='right' to closed='left', label='left'


# In[ ]:


ts.resample('5min', closed='left').sum()
ts.resample('5min', closed='right').sum()


# In[ ]:


ts.resample('5min', closed='left', label='left').sum()
ts.resample('5min', closed='right', label='right').sum()


# In[ ]:


ts.resample('5min', loffset='-1s').sum()


# #### # Open-High-Low-Close (OHLC) resampling
#   * 시가-고가-저가-종가

# In[ ]:


ts.resample('5min').ohlc()
# note: output changed because of changed defaults


# #### # Resampling with GroupBy

# In[ ]:


rng = pd.date_range('1/1/2017', periods=100, freq='D')
ts = Series(np.arange(100), index=rng)
ts
ts.index[0].month
ts.groupby(lambda x: x.month).mean()
ts.resample('M').mean()


# #### DatetimeIndex.weekday
#   - Monday=0, Sunday=6

# In[ ]:


ts.groupby(lambda x: x.weekday).mean()


# ### 10.6.2 Upsampling and interpolation

# In[ ]:


frame = DataFrame(np.random.randn(2, 4),
                  index=pd.date_range('9/1/2017', periods=2, freq='W-WED'),
                  columns=['Colorado', 'Texas', 'New York', 'Ohio'])
frame


# #### # 누락값 발생

# In[ ]:


df_daily = frame.resample('D').asfreq()
df_daily


# #### # 보간
#   * ffill(): 이전 값으로 채우기
#   * bfill(): 이후 값으로 채우기

# In[ ]:


frame.resample('D').ffill()


# In[ ]:


frame.resample('D').bfill()


# In[ ]:


frame.resample('D').ffill(limit=2)
frame.resample('D').bfill(limit=2)


# In[ ]:


frame.resample('W-THU').asfreq()
frame.resample('W-THU').ffill()
frame.resample('W-THU').bfill()


# ### 10.6.3 Resampling with periods

# In[ ]:


frame = DataFrame(np.random.randn(24, 4),
                  index=pd.period_range('1-2016', '12-2017', freq='M'),
                  columns=['Colorado', 'Texas', 'New York', 'Ohio'])
frame[:5]


# In[ ]:


annual_frame = frame.resample('A-DEC').mean()
annual_frame


# In[ ]:


# Q-DEC: Quarterly, year ending in December
annual_frame.resample('Q-DEC').ffill()
# note: output changed, default value changed from convention='end' to convention='start' + 'start' changed to span-like
# also the following cells


# In[ ]:


annual_frame.resample('Q-DEC', convention='end').ffill()


# In[ ]:


annual_frame
annual_frame.resample('Q-MAR').ffill()
annual_frame.resample('Q-MAR', convention='e').ffill()


# ## 10.7 Time series plotting

# In[ ]:


close_px_all = pd.read_csv('ch09/stock_px.csv', parse_dates=True, index_col=0)


# In[ ]:


close_px = close_px_all[['AAPL', 'MSFT', 'XOM']]; close_px


# In[ ]:


close_px = close_px.resample('B').ffill()


# In[ ]:


close_px
close_px.info()


# In[ ]:


close_px['AAPL'].plot()


# In[ ]:


import matplotlib
matplotlib.style.use('seaborn-whitegrid')
plt.rc('figure', figsize=(12, 4))
close_px.loc['2009'].plot();


# In[ ]:


close_px['AAPL'].loc['01-2011':'03-2011'].plot()


# In[ ]:


appl_q = close_px['AAPL'].resample('Q-DEC').ffill()
appl_q.loc['2009':].plot()


# ## 10.8 Moving window functions

# In[ ]:


close_px = close_px.asfreq('B').fillna(method='ffill')


# In[ ]:


close_px


# #### # 이동 합계

# In[ ]:


df_moving = DataFrame({'moving': np.random.randint(10, size=5)})
df_moving
df_moving.rolling(3).sum()
df_moving.rolling(3).sum().fillna(0)


# In[ ]:


df_moving2 = DataFrame({'moving2': np.random.randint(10, size=5)})
df_moving2
df_moving2.rolling(3, center=True).sum()


# #### # 확장창 함수(expanding window functions)

# In[ ]:


df_moving
df_moving.expanding(3).sum()


# In[ ]:


close_px.AAPL.plot()
close_px.AAPL.rolling(window=250).mean().plot()
#pd.rolling_mean(close_px.AAPL, 250).plot()


# In[ ]:


close_px.AAPL.plot()
close_px.AAPL.rolling(window=250, min_periods=100).mean().plot()


# In[ ]:


appl_std250 = close_px.AAPL.rolling(250, min_periods=10).std()
appl_std250[5:12]


# In[ ]:


appl_std250.plot()


# #### # 표준 이동창 함수, 표준 확장창 함수, 지수적 가중 이동창 함수
#   * https://pandas.pydata.org/pandas-docs/stable/api.html#standard-moving-window-functions

# In[ ]:


# Define expanding mean in terms of rolling_mean
#expanding_mean = lambda x: rolling_mean(x, len(x), min_periods=1)
expanding_mean = lambda x: x.rolling(len(x), min_periods=3).mean()
close_px.apply(expanding_mean)

# 이럴 필요없이 expanding 함수를 사용하자
close_px.expanding(3).mean()


# In[ ]:


#pd.rolling_mean(close_px, 60).plot(logy=True)
close_px.rolling(60).mean().plot(logy=True)


# In[ ]:


close_px.expanding().mean().plot(logy=True)
close_px.expanding(60).mean().plot(logy=True)


# ### 10.8.1 Exponentially-weighted functions
#   * 최근 값에 좀 더 많은 가중치를 두는 방법
#   * 균등 가중 방식에 비해 좀 더 빠르게 변화를 수용

# In[ ]:


# 애플사 주가의 60일 이동 평균과 EW의 60일 이동평균
fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True,
                         figsize=(12, 7))

aapl_px = close_px.AAPL['2005':'2009']

#ma60 = pd.rolling_mean(aapl_px, 60, min_periods=50)
ma60 = aapl_px.rolling(60, min_periods=50).mean()
#ewma60 = pd.ewma(aapl_px, span=60)
ewma60 = aapl_px.ewm(span=60).mean()

aapl_px.plot(style='k-', ax=axes[0])
ma60.plot(style='r--', ax=axes[0])
aapl_px.plot(style='k-', ax=axes[1])
ewma60.plot(style='b--', ax=axes[1])
axes[0].set_title('Simple MA')
axes[1].set_title('Exponentially-weighted MA')


# ### 10.8.2 Binary moving window functions
#   * 두 개의 시계열이 필요

# In[ ]:


close_px
spx_px = close_px_all['SPX']
spx_px


# In[ ]:


spx_rets = spx_px / spx_px.shift(1) - 1
returns = close_px.pct_change()
#corr = pd.rolling_corr(returns.AAPL, spx_rets, 125, min_periods=100)
corr = returns.AAPL.rolling(window=125, min_periods=100).corr(spx_rets)
corr.plot()


# In[ ]:


#corr = pd.rolling_corr(returns, spx_rets, 125, min_periods=100)
corr = returns.rolling(window=125, min_periods=100).corr(spx_rets)
corr.plot()


# ### 10.8.3 User-defined moving window functions
#   * 배열의 각 조각으로부터 단일 값(감소)을 반환해야 함

# In[ ]:


from scipy.stats import percentileofscore
score_at_2percent = lambda x: percentileofscore(x, 0.02)
#result = pd.rolling_apply(returns.AAPL, 250, score_at_2percent)
# 애플 사 주식의 수익 2%의 백분위 점수
result = returns.AAPL.rolling(window=250).apply(score_at_2percent)
result.plot()


# ## 10.9 Performance and Memory Usage Notes
#   * pandas의 노력
#     - 기존 시계열에 대한 뷰 생성
#     - 하위 빈도(일간 빈도 이상)에 대한 인덱스를 중앙 캐시에 저장
#     - 데이터 정렬 연산과 리샘플링의 고도 최적화

# In[ ]:


rng = pd.date_range('1/1/2000', periods=10000000, freq='10ms')
ts = Series(np.random.randn(len(rng)), index=rng)
ts


# In[ ]:


#ts.resample('15min', how='ohlc').info()
ts.resample('15min').ohlc().info()


# In[ ]:


get_ipython().run_line_magic('timeit', "ts.resample('15min').ohlc()")


# In[ ]:


rng = pd.date_range('1/1/2000', periods=10000000, freq='1s')
ts = Series(np.random.randn(len(rng)), index=rng)
get_ipython().run_line_magic('timeit', "ts.resample('15s').ohlc()")

