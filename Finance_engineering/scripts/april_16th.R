#볼록성
bondconv <- function(facevalue=1000, discrate = 10, maturity=10, couprate=8){
  faceval <- as.numeric(facevalue)
  discrate = as.numeric(discrate)/100
  maturity <- maturity
  
  freq <- 4
  times <- seq(from=0.25, by=0.25, length.out = maturity*freq)
  
  tsquare <- times^2
  
  pvfactors= exp(-discrate*times)
  
  coupon <- couprate*faceval/(100*freq)
  cashflows <- rep(coupon, maturity*freq)
  cashflows[length(cashflows)] = cashflows[length(cashflows)]+faceval
  
  price <- sum(cashflows * pvfactors)
  conv = sum(cashflows * pvfactors * tsquare)/(price)
  conv <- round(conv,2)
  
  plot(1:10, 1:10, type="n", xlab="", ylab="", axes=FALSE, frame=TRUE)
  text(5,5,paste("볼록성: ", conv), cex=1.4)
}

bondconv(1000, 10, 10, 8)


#듀레이션과 만기의 관계
#듀레이션과 만기의 관계
durmaturity <- function(maxmaturity=30, faceval=1000, couprate=10, disctate=20){
  
  maxmaturity <- 30
  faceval <- 1000
  
  bonddur = function(couprate=10, discrate=20, maturity, faceval=1000){
    couprate <- as.numeric(couprate)
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





#전환사채 가격 결정 예시
#3.0.3으로 다운그레이드 해야함
install.packages("drat")
drat::addRepo("ghrr")
options("repos")


install.packages("RQuantLib", type="binary")
library(RQuantLib)


today <- Sys.Date()

#거래 날짜와 정산 날짜 정하고 평행 수익률 곡선이 주어진 상태에서 할인률 곡선의 가치 계산
params <- list(tradeDate = today - 2, settleDate=today, dt = 0.25)
times <- seq(0, 10, 0.1)
dividendYield <- DiscountCurve(params, list(flat=10e-6), times)
riskFreeRate <- DiscountCurve(params, list(flat=0.05), times)

process <- list(underlying=20, divYield = dividendYield, rff=riskFreeRate, volatility=0.2)


#채권을 주식으로 변환한다고 결정시, 공통주 갖는 변환 비율 명시
bondparams <- list(exercise = "eu", faceAmount=100, redempion=100, creditSpread=0.02, conversionRatio=4, issueDate=as.Date(today+2), maturityDate=as.Date(today+1825))

#연 쿠폰 지급
dateparams <- list(settlementDays=3, dayCounter="ActualActual", period="Annual", businessDayConvention="Unadjusted")

ConvertibleFixedCouponBond(bondparams, coupon=0.05, process, dateparams)
#위의 함수의 결과를 보면, 순현재가치가 107.1172, dirty price는 경과이자를 반영한 가격임. 현금흐름에 대해 보면, 1년에 5씩 4년간 현금흐름이 있음. 이표채이기 떄문임. 마지막해에 원금+이자


#기초자산의 변화
res <- sapply(seq(1,30,1), function(s){
  
  process$underlying = s
  ConvertibleFixedCouponBond(bondparams, coupon=0.5, process, dateparams)$NPV
})

plot(1:30, res, type="l", xlab="기초주식가격", ylab="변환채권가치")




##큐빅 스플라인 회귀
install.packages("termstrc")
library(termstrc)

#정부채권 내장 자료 사용
data(govbonds)
str(govbonds[['GERMANY']])

#채권 데이터를 현금 흐름, 만기, 채권 수익률, 듀레이션 기반 가중치 행렬, 경과 이자 및 이를 포함한 채권 가격을 산출하는 함수로 전처리
prepro <- prepro_bond('GERMANY', govbonds)

m <- prepro$m[[1]]

n <- ncol(m)
s <- round(sqrt(n))

c(floor(min(m[,1])), max(m[,ncol(m)]))

i <- 2:(s-2)
h <- trunc(((i-1) * n) / (s-2))
theta <- ((i-1) * n) / (s-2) - h

apply(as.matrix(m[, h]), 2, max) + theta * (apply(as.matrix(m[,h]), 2, max) - apply(as.matrix(m[, h]), 2 ,max))

c(floor(min(m[, 1])), apply(as.matrix(m[, h]), 2, max) + theta * (apply(as.matrix(m[, h+1]), 2, max)- apply(as.matrix(m[, h]), 2, max)), max(m[, ncol(m)]))

x <- estim_cs(govbonds, 'GERMANY')
x$knotpoints[[1]]

plot(x)
par(mfrow=c(2,1))
plot(x$discount, multiple= TRUE)
plot(x$forward, multiple = TRUE)



