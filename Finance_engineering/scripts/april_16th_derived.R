#블랙숄즈 모델
install.packages("fOptions")
library(fOptions)

#r : 무위험금리, 변동성 : sigma, 재고 수행 비용 : b
GBSOption(TypeFlag = "c", S=900, X=950, Time= 1/4, r = 0.02, sigma = 0.22, b = 0.02)

GBSOption(TypeFlag = "p", S=900, X=950, Time= 1/4, r = 0.02, sigma = 0.22, b = 0.02)@price


#이항트리모델
CRRBinomialTreeOption(TypeFlag = "ce", S=900, X=950, Time= 1/4, r= 0.02, b= 0.02, sigma=0.22, n=3)@price

CRRBinomialTreeOption(TypeFlag = "pe", S=900, X=950, Time= 1/4, r= 0.02, b= 0.02, sigma=0.22, n=3)@price


par(mfrow=c(1,1))
CRRTree <- BinomialTreeOption(TypeFlag = "ce", S=900, X=950, Time=1/4, r=0.02, b=0.02, sigma=0.22, n=3)
BinomialTreePlot(CRRTree, dy=1, xlab="시간 단계", ylab="상승 단계", xlim = c(0,4))
title(main="콜 옵션 트리")



prices <- sapply(1:200, function(n){
  CRRBinomialTreeOption(TypeFlag = "ce", S=900, X=950, Time=1/4, r= 0.02, b=0.02, sigma=0.22, n=n)@price
})

str(prices)

price <- GBSOption(TypeFlag = "c", S=900, X=950, Time=1/4, r=0.02, sigma=0.22, b=0.02)@price
plot(1:200, prices, type='l', xlab="단계 개수", ylab="가격")
abline(h=price, col='red')
legend("bottomright", legend = c('CRR-가격', 'BS-가격'), col=c('black', 'red'), pch=19)




sapply(c('delta', 'gamma', 'vega', 'theta', 'rho'), function(greek){
  GBSGreeks(Selection=greek, TypeFlag = "c", S=900, X=950, Time=1/4, r=0.02, b=0.02, sigma=0.22)
})


deltas <- sapply(c(1/4, 1/20, 1/50), function(t){
  sapply(500:1500, function(S){
    GBSGreeks(Selection = 'delta', TypeFlag = "c", S=900, X=950, Time=t, r=0.02, b=0.02, sigma=0.22)
  })
})

plot(500:1500, deltas[,1], ylab='Delta of call option', xlab="Price of the underlying (S)", type='l')
lines(500:1500, deltas[,2], col='blue')
lines(500:1500, deltas[,3], col='red')

legend("bottomright", legend=c('t=1/4', 't=1/20', 't=1/50'), col= c('black', 'blue', 'red'), pch=19)


straddles <- sapply(c('c', 'p'), function(type){
  sapply(500:1500, function(S){
    GBSGreeks(Selection='delta', TypeFlag = type, S=S, X=950, Time=1/4, r=0.02, b=0.02, sigma=0.22)
  })
})

plot(500:1500, rowSums(straddles), type='l', xlab="Price of the underlying (S)", ylab='Delta of straddle')


goog <- read.csv('goog_calls.csv')
volatilites <- sapply(seq_along(goog$Strike), function(i){
  GBSVolatility(price=goog$Ask.Price[i], TypeFlag = "c", S=866.2, X=goog$Strike[i], Time= 88/360, r=0.02, b=0.02)
})


str(volatilites)

plot(x = goog$Strike, volatilites, type='p', ylab="내재변동성", xlab="행사가격 (x)")


