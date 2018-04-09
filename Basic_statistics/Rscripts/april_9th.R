library(readxl)

library(psych)

#일원분산분석
anova <- read_excel("C:/R/자료/source (2)/준비파일/fisher.xlsx")

group <- as.character(anova$낚시터)
boxplot(anova$물고기~ group)
bartlett.test(anova$물고기~ group)
with(anova, tapply(물고기, group, mean))
with(anova, tapply(물고기, group, var))

#linear model 함수사용
out1=lm(anova$물고기~ group)

#F값과 p-value에 따라 낚시터에 따라 잡히는 물고기 비율이 다르다는 것을 확인할 수 있다.(대립가설 채택)
summary(out1)

#Analysis of variance model, lm과 같은 용도로 사용
out2=aov(formula = anova$물고기~ group)
summary(out2)
#같은 결론에 도달

out3=anova(out1)
out3
#TukeyHSD(out1)

#p adj는 조정된 pvalue를 의미
TukeyHSD(out2)


#이원분산분석 
anova2 <- read_excel("C:/R/자료/source/이원분산분석.xls")
anova2
group1 <- as.character(anova2$흡연석여부)
group2 <- as.character(anova2$위치)
boxplot(anova2$매출액 ~ group1)
boxplot(anova2$매출액 ~ group2)

with(anova, tapply(anova2$매출액, group1, mean))
out1=aov(anova2$매출액~ group1 + group2 + group1*group2)

#각각의 연관성에 따라 상호작용 효과를 분석함.
anova(out1)

out2=aov(anova2$매출액~ group1 + group2)
anova(out2)
TukeyHSD(out1)

#상관분석

#산점도 예

plot(dist ~ speed, data=cars, type="p", pch=20, col ="blue")
#공분산
cov(cars$dist, cars$speed)

#단위 조정을 위해 상관계수로 표준화
#0~1 사이이므로 양의 상관관계가 있다고 확인가능함.
cor(cars$dist, cars$speed)

#상관분석 함수(기본은 pearson)
cor.test(~dist+speed, data=cars, method= c("pearson"))

#p-value가 p-value = 1.49e-12로 매우 작게 나왔으므로, 관계가 있다는 것을 알 수 있다. 귀무가설이 상관관계 0 이었음.

##광고비가 매출에 영향을 미치는지 분석하기
#상관분석
ad <- read_excel("C:/R/자료/source/단순 회귀분석.xls")

plot(매출액 ~ 광고비, data=ad, type="p", pch=20, col= "blue")
cov(ad$매출액, ad$광고비)
cor(ad$매출액, ad$광고비)
cor.test(~매출액+광고비, data=ad, method = c("pearson"))


##교차분석
vacation <- c(68, 32)
prob <- c(0.5, 0.5)
chisq.test(vacation, p=prob)

#둘의 선호도가 같다고 귀무가설을 두어지만 p값을 통해 바다와 산의 선호도가 다르게 나타난다는 것을 확인할 수 있다. 대립가설 채택

#교차분석 다른 예제
#멘델의 유전법칙을 알아보기 위해 완두콩을 재배한 결과 얻은 수확이 9:3:3:1 의 4가지 완두콩 형질에 부합하는지 확인(총 556종)


#관측빈도를 mendel에 넣음
mendel <- c(315, 101, 108, 32)

#기대확률은 9:3:3:1의 비율임.
prob <- c(9, 3, 3, 1) /16

chisq.test(mendel, p=prob)


purchasing <- read_excel("C:/R/자료/source/교차_카이제곱.xls")
purchasing.table <- xtabs(~구매의사+지역, data=purchasing)
purchasing.table

chisq.test(purchasing.table)


##단순회귀분석

ad <- read_excel("C:/R/자료/source/단순 회귀분석.xls")
ad

plot(매출액 ~ 광고비, data =ad, type="p", pch= 20, col="blue")


#회귀 계수 추정
reg=lm(ad$매출액 ~ ad$광고비)
reg
summary(reg)
#summary에 따라 광고비 증가에 따라 매출액 0.81증가라고 추정
#t value는 t검정을 사용하였기에 t통계량이 나온 것.
#pr(>|t|)의 p-value를 통해 매우 유의하다는 것을 알 수 있음.
#결정계수는 0.6914로 대략 69%가 설명된다고 확인가능함.
#조정된 결정계수는 독립변수가 현재 1개이므로 큰 문제 없음. 독립변수가 증가하는 것만으로 설명력에 관계없이 R^2값이 증가하는 경우가 있어서 이를 조정해주는 값임


#추정한 회귀계수에 대해 신뢰구간 확인(기본값 95%)
confint(reg)


#친절도, 재구매 회귀분석 예제

a <- read_excel("C:/R/자료/source (2)/준비파일/친절도재구매.xlsx")
plot(재구매 ~ 친절도, data = a, type="p", pch=20, col="blue")
reg0=lm(a$재구매 ~ a$친절도)
reg0

summary(reg0)

confint(reg0)

#잔차의 독립성 2근처면 독립, 1.4보다 작으면 양의 상관관계 
library(car)
res=residuals(reg)

#더빈왓슨 테스트는 귀무가설이 0, 대립가설이 양의 상관관계를 의미. 따라서 2 근처면 독립 즉, 잔차가 독립이며, 1.4보다 작으면 양의 상관관계를 갖는다고 판
durbinWatsonTest(res)

par(mfrow=c(2,2))


#Residuals vs Fitted는 잔차의 독립성 테스트, Normal Q-Q는 정규성 테스트. 45도를 따르면 정규분포를 따른다고 판단 
plot(reg)

#정규성검정
shapiro.test(res)

#산점도와 회귀직선
plot(ad$매출액 ~ ad$광고비, cex=1, lwd=2)
#회귀선
abline(reg, lwd=2, col="red" )


##다중회귀분석
phone <- read_excel("C:/R/자료/source/다중 회귀분석_요인저장_변수 계산.xls")
reg1=lm(phone$만족감 ~ phone$외관 + phone$유용성 + phone$편의성)
reg1
summary(reg1)
confint(reg1)


par(mfrow=c(2,2))
plot(reg1)

#정규성검정
res1=residuals(reg1)
shapiro.test(res1)

#잔차 독립성검정
durbinWatsonTest(res1)

#다중 공선성 검정
vif=vif(reg1)
vif



##다중회귀분석 다른 예제
a <- read_excel("C:/R/자료/source (2)/준비파일/친절도재구매사은품.xlsx")

reg1=lm(a$재구매 ~ a$친절도 + a$사은품)
reg1

#p-value에 따라 회귀식이 유의하며 설명력이 29퍼센트라는 것을 확인할 수 있다.
summary(reg1)
confint(reg1)

#일정한 가정들을 실제 만족하는지 확인
#잔차 독립, 정규성 확인
par(mfrow=c(2,2))
plot(reg1)

res1=residuals(reg1)
#정규성검정
shapiro.test(res1) #정규성 가지고 있음을 확인 가능

#잔차독립성 검정
durbinWatsonTest(res1) #1.4보다 작아서 양의 상관관계를 가진다고 볼 수 있다. 귀무가설 기각

#다중공선성 검정
#10보다 작으면 다중공선성이 없는 것으로 해석.
vif=vif(reg1)
vif

