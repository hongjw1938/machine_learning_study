#구간추정
#t.test사용
#t.test(x, y=NULL, alternative = c("two.sided"), "less", "greater"), mu=0, paired = FALSE, var.equal =FALSE, conf.level = 0.95 , ...)

#t.test를 이용한 신뢰구간 추출

h <- c(168, 160, 170, 162, 168, 163, 164, 167, 175, 179, 161, 155)

#conf.level을 이용해 신뢰구간 변경 가능
#confidence interval을 통해 신뢰구간을 확인할 수 있음.
t.test(h, alternative = "less",conf.level = 0.9)
t.test(h)
t.test(h, conf.level = 0.99)

#모평균의 구간추정 예제
#이승엽이 프로에 데뷔한 첫해부터 2014년까지의 홈런기록을 통해 2015년도에는 홈런을 몇 개 칠 수 있을 지 90,95,99 신뢰구간에서 예측 후 실제 홈런 개수와 비교해보기
#2015년 실제 26개를 쳤음.
homerun <- c(13, 9, 32, 38, 54, 36, 39, 47, 56, 14, 30, 41, 30, 8, 16, 5, 15, 21, 13, 32)
mean(homerun)
sqrt(var(homerun))

t.test(homerun, conf.level = 0.9)
t.test(homerun)
t.test(homerun, conf.level = 0.99)


#binom.test / prop.test --> correct true라면 연속성 반영
#주유소 77곳에서 용량을 계량, 그 결과 5곳의 주유소에서 주유해야할 용량과 실제 주유된 용량에 차이가 있었음. 전체 주유소에서 용량의 차이가 있을 비율 구하기.
#이산확률을 따른다 생각하고 95%신뢰구간에서 추정
binom.test(5, 77)
#정규 근사시(표본이 큰 경우)
prop.test(5, 77)


#프리미엄급 우유의 가격은 일반에 비해 약 20%이상 비쌈. 프리미엄급 우유를 구매한 경험이 이는 소비자에 일반 우유의 성분과 비고해 성분 차이를 알려주고, 프리미엄급 우유를 재구매할 것인지에 대해 50명의 표본 조사, 그 겨로가 4명의 선택이 달라졌을 때, 전체 우유 소비자의 90% 신뢰구간으로 프리미엄 우유에 대한 선택이 달라질 비율의 신뢰구간 구하기.
binom.test(4, 50, conf.level = 0.9)
prop.test(4, 50, conf.level = 0.9)



#install.packages("EnvStats")
library(EnvStats)
varTest(trees$Volume)

#이승엽 선수의 홈런 분산을 varTest로 추정해보기
varTest(homerun)



#모집단 평균의 가설검정
#음료수 나트륨 함량을 15개 표본을 대상으로 살펴보고자 함. 가설을 수립하고 유의수준 0.05에서 검정을 실시.
n <- c(308,302,290,292,327,290,320,285,315,285,295,288,310,325,300)

t.test(n, mu=300)
#단측 검정시.
t.test(n, mu=300, alternative = "less")
t.test(n, mu=300, alternative = "greater")

#p-value가 0.05보다 크기 때문에 귀무가설을 기각할 수 없다.


library(readxl)

#대응표본 t검정
diet <- read_excel("C:/Users/student/Desktop/R/통계학/자료/source/대응표본 t 검정.xls")
t.test(diet$`복용 전`, diet$`복용 후`, paired=TRUE)


#두 모집단 간의 평균 차이에 대한 가설검정
#독립표본 t 검정
battery <- read_excel("C:/Users/student/Desktop/R/통계학/자료/source/독립표본 t 검정.xls")
#분산이 동일하다는 가정하이므로 var.equal은 TRUE로 테스트한다.
t.test(battery$`제조사`, battery$`작동시간`, var.equal=TRUE)
g1 <- as.matrix(battery)[,2][1:100]
g2 <- as.matrix(battery)[,2][100:200]
t.test(g1,g2,var.equal=TRUE)
