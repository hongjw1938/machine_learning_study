##cat은 유닉스 명령어 이므로 사용할 수 없다.
#!cat ch06/ex1.csv
!type ch06\ex1.csv

df = pd.read_csv('ch06/ex1.csv') #dataframe으로 읽어옴
df


pd.read_table('ch06/ex1.csv', sep=',') #구분자를 정해주면 table함수로도 읽을 수 있다.

#!cat ch06/ex2.csv
!type ch06\ex2.csv

pd.read_csv('ch06/ex2.csv', header=None) #header를 none으로 주면 rangeIndex로 컬럼 이름이 주어진다.
pd.read_csv('ch06/ex2.csv', names=['a', 'b', 'c', 'd', 'message'])

names = ['a', 'b', 'c', 'd', 'message']
pd.read_csv('ch06/ex2.csv', names=names, index_col='message') #message컬럼을 index column으로 사용함

#!cat ch06/csv_mindex.csv
!type ch06\csv_mindex.csv
parsed = pd.read_csv('ch06/csv_mindex.csv', index_col=['key1', 'key2']) #key를 계층으로 쌓음. list로 줘야함.
parsed

!type ch06\ex3.txt
list(open('ch06/ex3.txt')) #line을 하나씩 읽고 list에 담는다.

result = pd.read_table('ch06/ex3.txt', sep='\s+') #\s는 white space로 공백의 역할을 하는 모든 것을 의미.
                                                  # +는 선행문장이 0개 이상 포함되어 있는 패턴을 의미
result


#!cat ch06/ex4.csv
!type ch06\ex4.csv
pd.read_csv('ch06/ex4.csv', skiprows=[0, 2, 3]) #0, 2, 3번 row는 skip함.

#!cat ch06/ex5.csv
!type ch06\ex5.csv
result = pd.read_csv('ch06/ex5.csv')
result
pd.isnull(result)

result = pd.read_csv('ch06/ex5.csv', na_values=['NULL']) # NaN값으로 인식할 문자 = 'NULL'로 지정
result

sentinels = {'message': ['foo', 'NA'], 'something': ['two']} # message col의 foo / NA , something col의 two를 NaN으로
pd.read_csv('ch06/ex5.csv', na_values=sentinels)



##6.1.1 텍스트 파일 조금씩 읽어오기
result = pd.read_csv('ch06/ex6.csv')
result


pd.read_csv('ch06/ex6.csv', nrows=5) #5건만 읽음.

chunker = pd.read_csv('ch06/ex6.csv', chunksize=1000) # TextFileReader클래스의 객체가 됨.
                                                      # chunksize에 따라 분리된 파일을 순회할 수 있다.
chunker


chunker = pd.read_csv('ch06/ex6.csv', chunksize=1000)

tot = Series([])
for piece in chunker:
    #tot = 으로 다시 넣어주지 않으면 개별적인 chunk를 만들어서 반환함.
    tot = tot.add(piece['key'].value_counts(), fill_value=0) #각 조각의 key를 세서 그 값을 추가함. 없으면 0으로
                                                             #해당 값을 모아서 합침.

tot = tot.sort_values(ascending=False) # 가장 많이 나온 value 순서로 sort, ascending=False이므로 내림차순
tot

tot[:10]



##텍스트 형식으로 기록하기
data = pd.read_csv('ch06/ex5.csv')
data

data.to_csv('ch06/out.csv') #쉼표로 구분된 형식으로 변형(원본은 그대로)
#!cat ch06/out.csv
!type ch06\out.csv


data.to_csv(sys.stdout, sep='|') #분리 문자를 지정 , 표준출력

data.to_csv(sys.stdout, na_rep='NULL') 
data.to_csv(sys.stdout, index=False, header=False) #column, index를 제외

data.to_csv(sys.stdout, index=False, columns=['a', 'b', 'c']) #특정컬럼만 쓰기.
    
dates = pd.date_range('1/1/2000', periods=7) #날짜 range해서 시작날짜를 입력받고 periods에 기입된 수 만큼 만듦
dates

ts = Series(np.arange(7), index=dates)
ts

ts.to_csv('ch06/tseries.csv') # 저장
#!cat ch06/tseries.csv
!type ch06\tseries.csv

Series.from_csv('ch06/tseries.csv', parse_dates=True) #parse_dates=True로 하지 않으면 날짜를 문자열로 인식함.

ts1 = Series.from_csv('ch06/tseries.csv', parse_dates=True) #parse_dates=True로 하지 않으면 날짜를 문자열로 인식함.
ts1
ts1.index

ts2 = Series.from_csv('ch06/tseries.csv', parse_dates=False)
ts2.index #이 경우는 object타입인 것을 확인할 수 있다.


##수동으로 구분형식 처리하기
#!cat ch06/ex7.csv
!type ch06\ex7.csv

import csv
f = open('ch06/ex7.csv')

for line in reader:
    print(line)
    
    
lines = list(csv.reader(open('ch06/ex7.csv'))) #list에 의해서 내부적으로 반복이 일어남.
lines

data_dict = {h: v for h, v in zip(header, zip(*values))} # *values에 의해 values 중첩 list가 풀리게됨.
                                                         # * 는 여러개의 argument가 들어올 때 가변 인자로써 사용
                                                         # zip([1, 2, 3], [1, 2, 3, 4])와 같이 형성됨.
data_dict


#zip함수 리뷰
zip(['a', 'b', 'c'], ['d', 'e', 'f'])
for i in zip(['a', 'b', 'c'], ['d', 'e', 'f']):
    print(i)

# *args
r = [3, 10, 2]
list(range(*r))
#결과는 [3, 5, 7, 9]가 된다.
list(range(3, 10, 2)) # 좌측 코드와 같음.


class my_dialect(csv.Dialect): #csv.Dialect class 상속
    lineterminator = '\n' 
    delimiter = ';' #구분자
    quotechar = '"' #인용부호
    quoting = csv.QUOTE_MINIMAL
#위처럼 서브클래스를 정의하지 않고 csv.reader에 키워드 인자로 각 csv파일의 특징을 지정해 전달해도 된다.
#reader = csv.reader(f, delimiter='|')


#csv로 구분된 파일 기록
with open('mydata.csv', 'w') as f:
    writer = csv.writer(f, dialect=my_dialect)
    writer.writerow(('one', 'two', 'three'))
    writer.writerow(('1', '2', '3'))
    writer.writerow(('4', '5', '6'))
    writer.writerow(('7', '8', '9'))
    
    

##JSON

obj = """
{"name": "Wes",
 "places_lived": ["United States", "Spain", "Germany"],
 "pet": null,
 "siblings": [{"name": "Scott", "age": 25, "pet": "Zuko"},
              {"name": "Katie", "age": 33, "pet": "Cisco"}]
}
"""

import json
result = json.loads(obj) #python의 dict형식으로 읽음
result

asjson = json.dumps(result) #파이썬 객체를 JSON으로 변환
asjson

siblings = DataFrame(result['siblings'], columns=['name', 'age'])
siblings

siblings.to_json # json형태로 빠르게 쓸 수 있음. pandas가 제공함.


##웹스크래핑
from lxml.html import parse
#from urllib2 import urlopen 현재 좌측의 library는 사용불가
from urllib.request import urlopen

parsed = parse(urlopen('http://finance.yahoo.com/q/op?s=AAPL+Options'))

doc = parsed.getroot()
doc # Element html임.

links = doc.findall('.//a') # a태그를 전부 찾아줌.
len(links) #현재 297개의 a태그가 있음
links[15:20]

lnk = links[28]
lnk
lnk.get('href')
lnk.text_content()

urls = [lnk.get('href') for lnk in doc.findall('.//a')]
urls[-10:] #url들 반환



####6.4 데이터베이스와 함께 사용하기

import sqlite3

query = """
CREATE TABLE test
(a VARCHAR(20), b VARCHAR(20),
 c REAL,        d INTEGER
);"""

con = sqlite3.connect(':memory:')
con.execute(query)
con.commit()


data = [('Atlanta', 'Georgia', 1.25, 6),
        ('Tallahassee', 'Florida', 2.6, 3),
        ('Sacramento', 'California', 1.7, 5)]
stmt = "INSERT INTO test VALUES(?, ?, ?, ?)" #sql문장, Atlanta, Georgia, 1.25, 6이 순서대로 ?에 들어간다.

con.executemany(stmt, data) #여러개를 실행함.
con.commit()

cursor = con.execute('select * from test')
rows = cursor.fetchall() #전체를 불러옴.
rows

DataFrame(rows, columns=list(zip(*cursor.description))[0]) #컬럼의 이름을 지정


#pandas 모듈의 read_sql 함수를 이용해 select 쿼리문과 데이터베이스 연결객체만으로 간단하게 구현 가능하다.
import pandas.io.sql as sql
sql.read_sql('select * from test', con)


##oracle과 pandas

import numpy as np
from pandas import DataFrame, Series
import pandas as pd

from sqlalchemy import create_engine
import cx_Oracle


engine = create_engine('oracle://scott:tiger@localhost:1521/xe')
# 또는
#engine = create_engine('oracle://dream01:catcher@70.12.50.50:1521/XE')


#with는 리소스 준비 및 반환할 때 사용함.
with engine.connect() as conn, conn.begin(): # connect를 하고 
    data = pd.read_sql_table('emp', conn)
    
data.columns