#install.packages("psych")
library(psych)

attach(trees)

#전체적인 요약을 보여줌.
describe(Volume)

#분포 및 통계량에 대한 요약정보 보여주는 plot
boxplot(Volume)

i <- c(300, 150, 0, -100)
A <- c(0.58, 0.87, 0.55, 0.05)
B <- c(0.55, 0.51, 0.45, 0.05)
ma = sum(i*A)
mb = sum(i*B)
ma
mb

vara = sum(A*(i-ma)^2)
varb = sum(B*(i-mb)^2)
vara
varb

#주식회사 수익 비교 예제
ac_local <- c(100, 200, 400)
ac_local_ex <- c(0.3, 0.4, 0.5)

country_field <- c(200, 400, 800)
country_field_ex <- c(0.2, 0.4, 0.6)

eco_possi <- c(0.7, 0.5, 0.2)

ac <- ac_local * ac_local_ex * eco_possi
ac

country <- country_field * country_field_ex * eco_possi
country

E_a <- sum(ac)
E_c <- sum(country)


var_a <- sum(eco_possi*(ac_local * ac_local_ex - E_a)^2)
var_c <- sum(eco_possi*(country_field * country_field_ex - E_c)^2)

#분포함수

#dnorm : 밀도함수
#pnorm : 누적 분포 함수
#qnorm : 분위수 함수
#rnorm : 난수 발생

#값, 평균, 표준편차
pnorm(4, 5.4, sqrt(1.5))
qnorm(.1, 5.4, sqrt(1.5))


##이항분포 예시
#확률 질량함수 dbinom
#동전 10번 던졌을 때, 앞면이 3번
dbinom(3, 10, 0.5)

#누적분포함수 pbinom
#동전 10번 던졌을 때, 앞면이 6번 이하
pbinom(6, 10, 0.5)
#큰 방향으로 확인하고 싶다면 FALSE로 변경
pbinom(6, 10, 0.5, lower.tail = FALSE) # P(X > 6)

#분위수 qbinom
#누적 확률이 0.3이상이 되는 x
qbinom(0.3, 10, 0.5)

#난수 rbinom
#0.5의 확률로 10번 시행할 때의 난수를 30개 리턴
rbinom(30, 10, 0.5)


#이항분포 예제
dbinom(2, 49, 0.1)

#포아송분포
#확률질량함수
dpois(7, 5) #P(x=7), mu=5

?dpois
#1주일 단위 휴대전화 놓고 출근 3회, 1회 이하로 놓고 갈 확률, 4~5회일 확률 각각 구하기
for0 <- dpois(0, 3)
for1 <- dpois(1, 3)
for0 + for1

for4 <- dpois(4, 3)
for5 <- dpois(5, 3)
for4+for5

#누적분포함수
?ppois
ppois(7, 5)
ppois(7, 5, lower.tail = FALSE)


#분위수
qpois(0.7, 5) #누적확률이 0.7이상 되는 x

#모의실험 30회 반복
rpois(30, 5)


#K법무법인 수임 변수 사건 1년 약 100건
#그 중 작년 5건 패소, 패소확률 이항 / 포아송으로 계산 어떤 확률 분포 이용이 더 유리할지 설명
dbinom(5, 100, 0.05)
dpois(5, 5)



#t분포
#df = 8, p(T <= 1.397)
pt(1.397, 8)

#P(-1.397 <= T <= 1.180)
pt(1.180, 8) - pt(-1.397, 8)

#P(T >= t) = 0.025
qt(0.025, 8, lower.tail = FALSE)


#카이제곱분포
#df = 8
#P(x^2 <= 15.507)
pchisq(15.507, 8)
#P(x^2 >= 2.180)
pchisq(2.180, 8, lower.tail=FALSE)
#P(1.647 <= x^2 <= 13.362)
pchisq(13.362, 8) - pchisq(1.647, 8)



#구간추정
#토익 점수 평균 : 500, 표준오차 : 100, 신뢰도 90/95/99에서 모수 추정
# X-z*SE <= u <= X+z*SE
#90% : 500-1.64*100 <= u <= 500+1.64*100
#95% : 500-1.96*100 <= u <= 500+1.96*100
#99% : 500-2.58*100 <= u <= 500+2.58*100

