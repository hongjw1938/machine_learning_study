import pandas_datareader.data as web
gs = web.DataReader("MSFT", "iex", "2014-01-01", "2016-03-06")

gs['volume'] != 0
ma5 = gs['close'].rolling(window=5).mean()
print(type(ma5))
print(ma5)