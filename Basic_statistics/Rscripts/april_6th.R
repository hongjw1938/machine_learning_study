library(readxl)

library(psych)



anova <- read_excel("C:/Users/student/Desktop/R/통계학/자료/source/일원분산분석.xls")

group <- as.character(anova$편의점)





#Bartlett Test == 분산의 동질성을 확인함.



boxplot(anova$만족도 ~ group)



?bartlett.test

bartlett.test(anova$만족도 ~ group)

#각 그룹의 평균 구하기
with(anova, tapply(`만족도`,group,mean))

with(anova, tapply(`만족도`,group,sd))



out1 = lm(anova$만족도 ~ group)

#Intercept 는 기준이 되는 부분임.



out1

summary(out1)

#R-squared는 

#Adjusted R-squared

#F-statistitc은 



out2=aov(anova$만족도 ~group)

out2

#Residualis는 잔차임



out3 = anova(out1)

out3



TukeyHSD(out2)



anova <- read_excel("C:/Users/student/Desktop/R/통계학/자료/source (2)/준비파일/연습문제.xlsx")

group <- as.character(anova$A낚시터)