library(Quandl)
?Quandl.auth
Quandl.api_key("yourauthenticationtoken")
G <- Quandl('GOOG/NASDAQ_GOOG', start_date='2014-04-01', end_date='2016-02-19')

str(G)


SP500 <- Quandl('YAHOO/INDEX_GSPC', start_date='2014-04-01', end_date='2016-02-19')
SP500 <- SP500$'Adjusted Close'

LIBOR <- Quandl('FED/RILSPDEPM01_N_B', start_date='2014-04-01', end_date='2016-02-19')
LIBOR <- LIBOR$Value

sapply(list(G, SP500, LIBOR), length)

cdates <- Reduce(intersect, list(G$Date, SP500$Date, LIBOR$Date))

G <- G[G$Date %in% cdates, 'Close']

#위의 데이터를 가져올 때 network상 API_KEY의 인증에 문제가 있어 다른 방법 사용
#install.packages("quantmod")
library(quantmod)
??getSymbols
G <- getSymbols("GOOG", env=NULL, from='2014-04-01', to='2016-02-19')

SP500 <- getSymbols("^GSPC", env=NULL, from='2014-04-01', to='2016-02-19')

#해당 방법도 통하지 않으면 엑셀 파일을 직접 읽는 방식을 채택
getwd()
G <- read.csv("Finance_engineering/data/GOOG.csv")
SP500 <- read.csv("Finance_engineering/data/SP500.csv")
LIBOR <- read.csv("Finance_engineering/data/LIBOR.csv")
LIBOR
sapply(list(G, SP500, LIBOR), length)

G <- Quandl('GOOG/NASDAQ_GOOG', start_date='2014-04-01', end_date='2016-02-19')
SP500 <- Quandl('YAHOO/INDEX_GSPC', start_date='2014-04-01', end_date='2016-02-19')
LIBOR <- Quandl('FED/RILSPDEPM01_N_B', start_date='2014-04-01', end_date='2016-02-19')

cdates <- Reduce(intersect, list(G$Date, SP500$Date, LIBOR$Date))

G <- G[G$Date %in% cdates, 'Close']
SP500 <- SP500[SP500$Date %in% cdates, 'Adj Close']
LIBOR <- LIBOR[LIBOR %in% cdates, 'Value']

logreturn <- function(x) log(tail(x, -1) / head(x, -1))

rft <- log(1 + head(LIBOR, -1)/ 36000 * diff(cdates))
str(rft)


#간단한 베타 추정

cov(logreturn(G) - rft, logreturn(SP500)- rft) / var(logreturn(SP500) - rft)

riskpremium <- function(x) logreturn(x) - rft
cov(riskpremium(G), riskpremium(SP500)) / var(riskpremium(SP500))


#선형 회귀식 베타 추정
fit <- lm(riskpremium(G) - riskpremium(SP500))

lm(formula = riskpremium(G) ~ riskpremium(SP500))


plot(riskpremium(SP500), riskpremium(G))
abline(fir, col = 'red')

fit <- lm(riskpremium(G) ~ -1 + riskpremium(SP500))

summary(fit)

summary(lm(riskpremium(G) ~ riskpremium(SP500)))

par(mforw = c(2,2))
plot(fit)


#모델검정은 교재 확인

##채권
#고정금리채권의 시장 리스크 측정
#액면가 미화 1000달러인 10년 만기 채권의 쿠폰 비율은 연 8%이며, 분기마다 지급, 이 채권의 수익률 곡선은 10%의 무한 복리 이자율을 가질 때 수평형의 모습을 보임. 
#액면가는 bondprice함수를 이용해 867.28달러로 구해짐. priceyield함수는 할인율과 채권 가격의 반비례 관계를 보여줌. 채권의 듀레이션은 bonddur를 사용
#구현

#############################################
bondprice <- function(facevalue=1000, couprate=8, discrate=10, maturity=10, frequency="quarterly", ratefreq="continuous comp"){
  faceval <- as.numeric(facevalue)
  discrate = as.numeric(discrate)/100
  maturity <- maturity
  
  if(frequency == "quarterly"){
    freq <- 4
    times <- seq(from=0.25, by=0.25, length.out=maturity*freq)
  } else if(frequency == "semi-annual"){
    freq <- 2
    times <- seq(from=0.5, by=0.5, length.out = maturity*freq)
  } else {
    freq <- 1
    times <- seq(from=1, by=1, length.out = maturity*freq)
  }
  
  if(ratefreq =="continuous comp"){
    pvfactors = exp(-discrate*times)
  } else if (ratefreq=="annual comp"){
    pvfactors = 1/(1+discrate)^times
  } else {
    pvfactors = 1/(1+discrate/freq)^(freq*times)
  }
  
  coupon <- couprate*faceval / (100*freq)
  cashflows <- rep(coupon, maturity*freq)
  cashflows[length(cashflows)]= cashflows[length(cashflows)]+faceval
  price <- sum(cashflows*pvfactors)
  
  price <- round(price,2)
  plot(1:10, 1:10, type="n", xlab="", ylab="", axes=FALSE, frame=TRUE)
  text(5,5, paste("Price: ", price), cex=1.4)
}

bondprice(1000,8,10,10,"quarterly", "continuous comp")


#############################################
priceyield <- function(maturity=10, frequency="quarterly", couprate=8){
  faceval <- 1000
  discrates =seq(0,30,1)
  prices = seq(0,0.30, length.out=length(discrates))
  
  bondvalues <- function(maturity, frequency, couprate){
    maturity <- maturity
    if (frequency == "quarterly"){
      freq <- 4
    } else if (frequency == "semi-annual"){
      freq <- 2
    } else {
      freq <- 1
    }
    coupon <- couprate*faceval / (100*freq)
    for(i in 1:length(discrates)){
      effrate = discrates[i] / (100*freq)
      effperiods = freq*maturity
      pv_coupons <- (coupon/effrate) * (1-(1+effrate)^(-effperiods))
      pv_face <- faceval * (1+effrate)^(-effperiods)
      prices[i] <- pv_coupons+pv_face
    }
    
    return (prices)
  }
  
  
  
  plot(discrates, bondvalues(maturity, frequency, couprate), xlab="Yield(%)", ylab="Bond Price",,type="l", lwd=2)
  
  title(paste("Par Value= ", faceval, "\nCoupon=", couprate, ", Maturity = ", maturity, "\n"))
}

priceyield(10, "quarterly", 8)

##############################################
bonddur <- function(facevalue=1000, discrate=10, maturity=10, couprate=8){
  facevalue=1000;
  discrate=10;
  maturity=10;
  couprate=8
  
  faceval <- as.numeric(facevalue)
  discrate = as.numeric(discrate)/100
  maturity <- maturity
  
  freq <- 4
  times <- seq(from=0.25, by=0.25, length.out=maturity*freq)
  pvfactors=exp(-discrate*times)
  
  coupon <- couprate*faceval/(100*freq)
  cashflows <- rep(coupon, maturity*freq)
  cashflows[length(cashflows)]=cashflows[length(cashflows)]+faceval
  price <- sum(cashflows*pvfactors)
  dur = sum(cashflows*pvfactors*times) / price
  dur <- round(dur,2)
  
  plot(1:10, 1:10, type="n", xlab="", ylab="",
       axes=FALSE, frame=TRUE)
  text(5, 5, paste("Duration: ", dur), cex=1.4)
}

bonddur



##############################################
bondconv <- function(facevalue=1000, discrate=10, maturity=10, couprate=8){
  faceval <- as.numeric(facevalue)
  discrate = as.numeric(discrate)/100
  maturity <- maturity
  
  freq <- 4
  times <- seq(from=0.25, by=0.25, length.out=maturity*freq)
  
  tsquare <- times^2
  pvfactors = exp(-discrate* times)
  
  coupon <- couprate*faceval/(100*freq)
  cashflows <- rep(coupon, maturity*freq)
  cashflows[length(cashflows)] = cashflows[length(cashflows)]+faceval
  
  price <- sum(cashflows*pvfactors)
  conv = sum(cashflows*pvfactors*tsquare)/(price)
  plot(1:10, 1:10, type="n", xlab="", ylab="", axes=FALSE, frame=TRUE)
  text(5, 5, paste("볼록성 :", conv),cex=1.4)
}
bondconv(1000,10,10,8)


#듀레이션과 만기의 관계
durmaturity <- function(maxmaturity=30, faceval=1000, couprate=10, disctate=20){
  
  maxmaturity <- 30
  faceval <- 1000
  
  bonddur = function(couprate=10, discrate=20, maturity, faceval=1000){
    couprate <- as.numeric(couprate)/100
    discrate = as.numeric(discrate)/100
    freq <- 1
    times <- seq(from=1, by=1, length.out=maturity*freq)
    
    pvfactors= 1/(1+discrate/freq)^(freq*times)
    
    coupon <- couprate*faceval/100
    cashflows <- rep(coupon,maturity*freq)
    cashflows[length(cashflows)] = cashflows[length(cashflows)]+faceval
    
    price <- sum(cashflows*pvfactors)
    dur = sum(cashflows*pvfactors*times)/price
    dur <- round(dur,2)
    
    return(dur);
  }
  
  bonddurs <- rep(0,maxmaturity)
  for(maturity in 1:maxmaturity){
    bonddurs[maturity]=bonddur(maturity=maturity,faceval=faceval)
  }
  
  
  plot(1:maxmaturity, bonddurs, type="l", lwd=2, ylim=c(1,maxmaturity), xlab="만기", ylab="듀레이션", frame=TRUE)
  
  title(paste("듀레이션 만기"))
}

durmaturity(30, 1000, 10, 20)
