#install.packages("readxl")
library(readxl)

student <- read_excel("C:/Users/student/Desktop/R/통계학/자료/source/빈도분석.xls")
student
attach(student)

#install.packages("descr")
library(descr)

#도수분포표
freq(성별, plot=FALSE)
freq(대학, plot=FALSE)
freq(대학)


#pie(대학)

#attach를 통해 특정 자료를 사용할 수 있음.
attach(trees)
hist(Volume)

#줄기잎
stem(Volume)


#중심경향도
mean(Volume)
median(Volume)
x <- table(Volume)
x

#최빈값의 name
names(which.max(x))

#4분위수, 최소/최대값 
summary(Volume)

#분산
var(x)


#신장 정보 파일
height <- read_excel("C:/Users/student/Desktop/R/통계학/자료/source (2)/준비파일/3장_신장.xlsx")

class_a <- height[height$반=='A반', ]
class_a <- as.numeric(class_a[2:11])
class_a
mean(class_a)

