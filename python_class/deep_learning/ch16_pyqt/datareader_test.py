import pandas_datareader.data as web
from datetime import datetime


def pushButtonClicked():
    code = "MSFT"
    df = web.DataReader(code, "iex", datetime(2017, 1, 1), datetime(2018, 5, 30))  # 날짜 반드시 명시할 것.

    print(df)
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA60'] = df['close'].rolling(window=60).mean()

pushButtonClicked()