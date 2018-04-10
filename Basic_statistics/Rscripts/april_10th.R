#R basic

#seq()
seq(from=1, to=5, by=2)
seq(1, 5, by=2)
seq(0, 1, by=0.001)
seq(0, 1, length.out = 1000)

#rep()

rep(c(1,2,3), times=2)
rep(c(1,2,3), each=2)

#vector
x <- c(1,2,3,4,5)
x[1]
x[c(1,2,3)]
x[-c(1,2,3)]

#length()
length(x)

#sum()
x <- c(1,2,3,4)
y <- c(5,6,7,8)
z <- c(1,2)
w <- c(1,2,3)
2+x
x+y
x+z
x+w

#is, as
x <- c(1,2,3,4)
is.vector(x)
f <- factor(x)
f
is.vector(f)
v <- as.vector(f)
f
v    #문자열로 이루어진 벡터

#all(), any()
x <- 1:5
x>3
all(x>3) #false
any(x>3) #true

#names() -- R의 객체를 불러오거나 지정하는 함수
height <- c(80, 90, 70, 170)
names(height) <- c("A", "B","C","D")
height


#head(), tail()
x <- 1:100
head(x, n=7)
tail(x)

#sample()
sample(10)
sample(45, 6)
sample(10, 3, replace = TRUE)
sample(10, 3, prob=(1:10)/55)
x <- seq(0, 1, by=0.1)
x
sample(x,3)

#which(x, arr.ind=FALSE)
x <- c(2, 4, -1, 3)
which( x>2)
names(x) <- c("a", "b", "c", "d")
which(x > 2)

#array(data =NA, dim=length(data),)
arr <- array(1:3, c(2,4))
arr
arr[1,]
arr[,3]
dimnamearr <- list(c("A","B"), c("a", "b", "c", "d"))
arr2 <- array(1:3, c(2,4), dimnames = dimnamearr)
arr2


v1 <- c(1, 2, 3, 4)
v2 <- c(5, 6, 7, 8)
v3 <- c(9, 10, 11, 12)

# vector을 열 단위로 합침
cbind(v1, v2, v3)
# vector을 행 단위로 합침
rbind(v1, v2, v3)


# 요인(Factor)은 '범주형'자료를 저장함
# factor(x = character(), levels, labels = levels, exclude = NA, ordered = is.ordered(x), nmax = NA)
# exclude = NA : 사용하지 않을 값을 지정

x <- c(1, 2, 3, 4, 5)

# levels를 통해 자료중 1,2,3,4 네 개의 값만 요인으로 사용
factor(x, levels=c(1,2,3,4))
factor(x, levels=c(1,2,3,4), exclude=c(1,2))
factor(x, levels=c(1,2,3,4), exclude=c(1,2), ordered=TRUE)

?apply
# apply(X, MARGIN, FUN, ...)
# x: apply를 적용할 배열, MARGIN: 함수를 적용할 차원, FUN: 적용할 함수
m1 <- cbind(v1, v2, v3)
m1

# 행별로 평균을 출력
apply(m1, 1, mean)
# 열별로 평균을 출력
apply(m1, 2, mean)

v <- c(1,2,3,4)

x<- factor(v)
is.factor(x)
is.factor(v)
as.factor(v)
v

?tapply
#tapply
score <- c(92,90,82,88,78,64,82,90)
subject <- c("k","k","m","m","m","m","k","k")
tapply(score, subject, mean)

#data.frame
name <- c("철수", "영희", "길동")
age <- c(21,20,31)
gender <- factor(c("M","F","M"))
character <- data.frame(name,age,gender)
character$name
character[1,]
character[,2]
character[3,1]

#attach
head(airquality)
#Ozone

Ozone <- c(1,2,3)
attach(airquality)
#Ozone
Ozone[1:5]

detach(airquality)
#Ozone

#with() - 데이터 프레임에 함수를 수행
#attach없어도 가능. 일시적인 이용시 with사용
head(cars)
mean(cars$speed)
mean(cars$dist)

with(cars, mean(speed))
with(cars, mean(dist))


#subset()
subset(airquality, Temp > 80)
subset(airquality, Temp > 80, select = c(Ozone, Temp))
subset(airquality, Temp > 80, select = -c(Ozone, Temp))

#str
str(airquality)
str(na.omit(airquality))

#data frame은 모든 속성(열 요소)들의 크기가 일정
# List: 서로 다른 기본 자료형을 가질 수 있음/ 순서존재
title <- "My List"
ages <- c(31, 41, 21)
numbers <- matrix(1:9, nrow=3)
names <- c("Baby", "Gentle", "none")
listEx <- list(title, ages, numbers, names)
listEx

#
listEx2 <- list(title=title, age=ages, number=numbers, name=names)
listEx2
listEx2[[1]]
listEx2$title
listEx2$age
listEx2$number
listEx2$name

x <- list(c(1,2,3,4), c(3,2,1))
v <- c(1,2,3,4)
is.list(x)
is.list(v)
a.l <- as.list(v)
a.l


#lapply
x <- list(a = 1:10, beta = exp(-3:3), logic=c(TRUE, FALSE, FALSE, FALSE))
lapply(x, mean)
sapply(x, mean)


ex <- c(1,3,7, NA, 12)
ex < 10
ex[ex <10]
ex[ex %% 2 == 0]
ex[is.na(ex)]



?fivenum
fivenum(ex)


#ifelse

x <- c(6:-4)
options(digits=3)


sqrt(x)
sqrt(ifelse(x >=0, x, NA))

x <- c(1,2,3)
x <- factor(x)
if(is.factor(x)) {length(x)
  } else {sum(x)}

x <- c(1,2,3)
x <- list(x)

if(is.factor(x)){
  length(x)
} else if (is.integer(x)){
  sum(x)
} else {
  paste(x, "element")
}

i <- 20
repeat {
  if(i > 25){
    break
  } else {
    print(i)
    i <- i+1
  }
}


#구구단
dan <- 2
i <-2
while(i <10 ){
  times <- i * dan
  print(paste(dan, "X", i, " = ", times))
  i <- i+1
}

dan <- 9
for(i in 2:9){
  times <- i * dan
  print(paste(dan, "X", i, " = ", times))
}

#Plot
str(Puromycin)
PuroTrt <- subset(Puromycin, state=="treated")
PuroUnTrt <- subset(Puromycin, state=="untreated")


plot(rate ~ conc, data=PuroTrt)
plot(rate ~ conc, data=PuroUnTrt, type="p")
plot(rate ~ conc, data=PuroUnTrt, type="l")
plot(rate ~ conc, data=PuroUnTrt, type="b")
