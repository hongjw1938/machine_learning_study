import pandas_datareader.data as web
import pandas as pd

gs = web.DataReader("MSFT", "iex", "2014-01-01", "2016-03-06")
new_gs = gs[gs['volume'] !=0]

#ma5 = gs['Adj Close'].rolling(window=5).mean()
#print(ma5.tail(10))

#ma5 = gs['Adj Close'].rolling(window=5).mean()
#new_gs['MA5'] = ma5
#print(new_gs)

# m20, ma60, ma120
ma20 = gs['close'].rolling(window=20).mean()
ma60 = gs['close'].rolling(window=60).mean()
ma120 = gs['close'].rolling(window=120).mean()

new_gs['MA20'] = ma20
new_gs['MA60'] = ma60
new_gs['MA120'] = ma120

print(new_gs.tail(10))

import matplotlib.pyplot as plt
plt.plot(new_gs.index, new_gs['close'], label="Close")
#plt.plot(gs.index, gs['MA5'], label="MA5")
plt.plot(new_gs.index, new_gs['MA20'], label="MA20")
plt.plot(new_gs.index, new_gs['MA60'], label="MA60")
plt.plot(new_gs.index, new_gs['MA120'], label="MA120")



#legend
plt.legend(loc='best')
plt.grid()
plt.show()