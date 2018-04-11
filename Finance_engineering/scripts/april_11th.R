#시계열 자료 분석에 도움을 주는 패키지를 사용할 것.
#zoo, xts, timeSeries
library(zoo)
setwd("C:/R/자료/0933OS_Code/Chapter 1")
require(graphics)

#co2 내장자료를 가지고 분석.
#각 성분을 확인할 수 있다.
m <- decompose(co2)
m$figure
plot(m)

#Apple의 주식 변화에 대한 분석
aapl <- read.zoo("aapl.csv", sep=",", header=TRUE, format="%Y-%m-%d")

plot(aapl, main= "APPLE Closing Prices on NASDAQ", ylab="Price(Usd)", xlab="Date")

head(aapl)
tail(aapl)
aapl[which.max(aapl)]

##단순 수익률 계산
ret_simple <- diff(aapl) / lag(aapl, k= -1) * 100

summary(ret_simple)
#날짜 자료를 정제한 coredata만 출력
summary(coredata(ret_simple))

#단순 수익률 중 최저/최대 수익률
ret_simple[which.min(ret_simple)]
ret_simple[which.max(ret_simple)]

#상대도수 이해 - 히스토그램
hist(ret_simple, breaks=100, main="Histogram of Simple Returns", xlab="%")


#복리수익률
ret_cont <- diff(log(aapl)) * 100
ret_cont
summary(coredata(ret_cont))

#window 메소드를 활용한 시계열 자료 부분집합 산출
aapl_2013 <- window(aapl, start='2013-01-01', end='2013-12-31')
aapl_2013[which.max(aapl_2013)]


#Value-at-Risk 값
quantile(ret_simple, probs = 0.01) #일일 손실률 7%보다 낮을 확률은 1%, 년에 2.5번 정도 발생


##영국의 주택 가격 예측 
#install.packages("forecast")
library(forecast)

hp <- read.zoo("UKHP.csv", sep=",", header=TRUE, format ="%Y-%m", FUN=as.yearmon)
frequency(hp)               

#1기 전과의 차이, 분모는 1기전의 값
hp_ret <- diff(hp) / lag(hp, k = -1) * 100

#계절성 없다고 판단, stationary는 정상성 찾고 있음을 판단, ic는 아카이케 정보 척도 사용
mod <- auto.arima(hp_ret, stationary = TRUE,seasonal = FALSE, ic = "aic")

#결과로 AR(2)가 가장 잘 설명한다는 것을 확인가능, ar1, ar2로 두 가지 나왔기 때문
mod

#신뢰구간 확인
#5%의 레벨에서 추정 값이 유의미한 값을 갖고 있다고 판단. 이들의 신뢰구간이 0을 포함하지 않았기 때문이다.
confint(mod)

#모델 적합성 진단
#결과는 변동 클러스터가 보이지 않고, 잔차간 자기 상관성이 보이지 않는다. Ljung-Box테스트에 의해 p-value가 높다는 것을 확인하였으므로 자기 상관성들이 서로 독립적이라는 귀무가설 기각할 수 없다.
tsdiag(mod)


#모델과 실제 측정치를 비교. 그래프 이용
plot(mod$x, lty=1, main="UK house prices: raw data vs. fitted values", ylab="Return in percent", xlab = "Date")
lines(fitted(mod), lty=2, lwd=2, col="red")

#모델 정확도 확인
accuracy(mod)

#향후 3개월 수익률 예측
predict(mod, n.ahead = 3)
plot(forecast(mod))

##공적분 예제
#urca 라이브러리는 공적분 관계 추정 및 단위근 검정을 위한 메소드 제공
#install.packages("urca")
library(urca)

prices <- read.zoo("JetFuelHedging.csv", sep=",", FUN=as.yearmon, format = "%Y-%m", header = TRUE)

#+0은 절편을 0으로 둔다는 의미
#기울기가 0.89이면 난방 1단위에 제트 0.89단위 상승. 헤지를 위해선 1이 되어야 함.
simple_mod <- lm(diff(prices$JetFuel) ~ diff(prices$HeatingOil)+0)
summary(simple_mod)

plot(prices$JetFuel, main="Jet Fuel and Heating Oil Prices", xlab="Date", ylab="USD")
lines(prices$HeatingOil, col="red")

#단위근 검정
#검정통계량은 -1.1335이고 이를 검정하는 기준은 tau임. 1percent 유의 수준에서 임계치가 더 작은 -3.46이므로 기각될 수 없다.
#따라서 단위근이 있다는 귀무가설을 기각할 수 없고 확률적 추세를 가진다고 판단함.
jf_adf <- ur.df(prices$JetFuel, type="drift")
summary(jf_adf)

#난방유 가격 단위근 검정
ho_adf <- ur.df(prices$HeatingOil, type="drift")
summary(ho_adf)

#정태적 균형 모델 추정 및 잔차 정상성 검정
mod_static <- summary(lm(prices$JetFuel ~ prices$HeatingOil))
error <- residuals(mod_static)
error_cadf <- ur.df(error, type="none")
#임계치보다 작은 통계량 값이 나왔으므로 비정상성을 가정한 귀무가설 기각.
summary(error_cadf)

?lag
#오차수정모형
djf <- diff(prices$JetFuel)
dho <- diff(prices$HeatingOil)
error_lag <- lag(error, k=-1)
mod_ecm <- lm(djf ~ dho + error_lag +0)
summary(mod_ecm)


##변동성 모델
intc <- read.zoo("intc.csv", header=TRUE, sep=",", format="%Y-%m", FUN= as.yearmon)

plot(intc, main = "Monthly returns of Intel Corporation", xlab = "Date", ylab = "Return in percent")                 

#Ljung-Box 검정
Box.test(coredata(intc^2), type="Ljung-Box", lag=12)
#위를 통해 자기상관이 없음을 가정한 귀무가설을 기각할 수 있다.

#FinTS 패키지의 LM검정도 같은 결과
#install.packages("FinTS")
#library(FinTS)
#ArchTest(coredata(intc))

#월별 수익률을 모델화하기 위해서 ARCH / GARCH사용
#install.packages("rugarch")
library(rugarch)

intc_garch11_spec <- ugarchspec(variance.model = list(garchOrder = c(1,1)), mean.model =list(armaOrder = c(0,0)))
intc_garch11_spec

#최대 우도 방법 이용한 계수 fitting
intc_garch11_fit <- ugarchfit(spec = intc_garch11_spec, data = intc)
intc_garch11_fit

#모델 성능 확인 - 백테스팅
intc_garch11_roll <- ugarchroll(intc_garch11_spec, intc, n.start=120, refit.every=1, refit.window = "moving", solver = "hybrid", calculate.VaR = TRUE, VaR.alpha = 0.01, keep.coef = TRUE)

report(intc_garch11_roll, type="VaR", VaR.alpha = 0.01, conf.level = 0.99)


#백테스팅 결과 그래프 - 확인
intc_VaR <- zoo(intc_garch11_roll@forecast$VaR[, 1])
#index(intc_VaR) <- as.yearmon(rownames(intc_garch11_roll@forecast$VaR))

intc_actual <- zoo(intc_garch11_roll@forecast$VaR[, 2])
#index(intc_actual) <- as.yearmon(rownames(intc_garch11_roll@forecast$VaR))

plot(intc_actual, type="b", main="99% 1 Month VaR Backtesting", xlab="Date", ylab="Return/VaR in percent")
lines(intc_VaR, col="red")
legend("topright", inset=.05, c("Intel return", "VaR"), col= c("black","red"), lty=c(1,1))


#VaR값 예측
intc_garch11_fcst <- ugarchforecast(intc_garch11_fit, n.ahead = 12)
intc_garch11_fcst