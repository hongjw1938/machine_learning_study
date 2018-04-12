myfactorial <- function(x){
  fact <-1
  i <- x
  while(i > 1){
    fact <- fact + i
    i <- i -1
  }
}
print(myfactorial(5))

#분산효과 각 자산 표준편차 0.2, 0.3 인 경우
div <- seq(-0.2, 1.2, length=100)
rhos <- c(-1, -0.5, 0, 0.5, 1)
out <- matrix(0, nrow=100, ncol=5)
for(i in 1:5){
  out[,i] <- sapply(div, function(a){
    rho <- rhos[i]
    ans <- a^2*0.2^2 + (1-a)^2*0.3^2 + 2*a*(1-a)*rho*0.2*0.3
    return(ans)
  })
}
matplot(div, out, type='l')
nms <- c('rho=-1', 'rho=-0.5', 'rho=0.5', 'rho=1')
legend("topright", legend=nms, lty=1:5, col=1:5, bty="n")

head(letters,-1)
tail(letters,-1)

#포트폴리오 최적화 - 라그랑지 구현
minvariance <- function(assets, mu =0.005){
  return <- log(tail(assets, -1) / head(assets, -1))
  Q <- rbind(cov(return), rep(1, ncol(assets)),
             colMeans(return))
  Q <- cbind(Q, rbind(t(tail(Q,2)), matrix(0,2,2)))
  B <- c(rep(0, ncol(assets)), 1, mu)
  solve(Q,b)
}

#quandl의 it 주식가격 데이터 가져오기
#install.packages("Quandl")
library(Quandl)

IT <- Quandl('DAROCZI/IT', start_date='2014-04-01', end_date='2016-02-19')

str(IT)

#자산의 수익률
assets <- IT[,-1]

#로그수익률 계산
return <- log(tail(assets, -1) / head(assets, -1))

head(return)

#공분산행렬
Q <- rbind(cov(return), rep(1, ncol(assets)), colMeans(return))
round(Q, 5)
Q <- cbind(Q, rbind(t(tail(Q,2)), matrix(0, 2, 2)))
round(Q, 5)


mu <- 0.005
b <- c(rep(0, ncol(assets)), 1, mu)

b

solve(Q, b)

minvariance(IT[,-1])


#포트폴리오 투자선
frontier <- function(assets){
  return <- log(tail(assets, -1) / head(assets,-1))
  Q <- cov(return)
  n <- ncol(assets)
  r <- colMeans(return)
  Q1 <- rbind(Q, rep(1, n), r)
  Q1 <- cbind(Q1, rbind(t(tail(Q1, 2)), matrix(0,2,2)))
  rbase <- seq(min(r), max(r), length=100)
  s <- sapply(rbase, function(x){
    y <- head(solve(Q1, c(rep(0, n), 1, x)), n)
    y %*% Q %*% y
  })
  plot(s, rbase, xlab="Variance", ylab="Return")
}

frontier(assets)



#지금의 과정을 간단하게 fPortfolio패키지 사용해서 결과 보기
#install.packages("timeSeries")
library(timeSeries)

IT <- timeSeries(IT[,2:6], IT[,1])

log(lag(IT)/ IT)
IT_return <- returns(IT)

#install.packages("PerformanceAnalytics")
library(PerformanceAnalytics)

#수익률 그래프
chart.CumReturns(IT_return, legend.loc = 'topleft', main='')


#투자선 그래프
#install.packages("fPortfolio")
library(fPortfolio)
plot(portfolioFrontier(IT_return))

Spec = portfolioSpec()
setSolver(Spec) = "solveRshortExact"
Frontier <- portfolioFrontier(as.timeSeries(IT_return), Spec, constraints = "Short")
frontierPlot(Frontier, col=rep('orange', 2), pch=19)
monteCarloPoints(Frontier, mcSteps = 1000, cex=0.25, pch=19)
grid()


#접선 포트폴리오와 자본시장선
n <- 6
mu <- 0.005
Q <- cbind(cov(return), rep(0, n-1))
Q <- rbind(Q, rep(0, n))
rf <- 0.001
r <- c(colMeans(return), rf)

Q <- rbind(Q, rep(1, n), r)
Q <- cbind(Q, rbind(t(tail(Q, 2)), matrix(0, 2, 2)))
b <- c(rep(0,n), 1, mu)

round(Q, 6)
b

#비중(합이 1이 됨. 아핀 결합이므로 비중이 음수가 나올 수 있다.)
w <- solve(Q, b)
w <- head(w, -3)
w / sum(w)
?solve


Spec <- portfolioSpec()
setSolver(Spec) <- "solveRshortExact"
setTargetReturn(Spec) <- mean(colMeans(IT_return))
efficientPortfolio(IT_return, Spec, 'Short')
minvariancePortfolio(IT_return, Spec, 'short')
minriskPortfolio(IT_return, Spec)
maxreturnPortfolio(IT_return,Spec)

??portfolio
