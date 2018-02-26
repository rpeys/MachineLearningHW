setwd("~/Documents/Columbia_DSI/machinelearning/HW/Homework 2/")

#### read in data ####
#train
x.train <- read.csv("hw2-data/X_train.csv", header = F)
y.train <- (read.csv("hw2-data/y_train.csv", header = F))$V1

##test
x.test <- read.csv("hw2-data/X_test.csv", header = F)
y.test <- read.csv("hw2-data/y_test.csv", header = F)$V1

#### problems 2d&e - logistic regression ####

#set every yi=0 to yi=-1
y.train[y.train==0] <- -1;
y.test[y.test==0] <- -1;

#add dimensions of +1s to each data point. 
x.train <- mutate(x.train, "w0.dim" = 1)
x.train <- x.train[,c("w0.dim", setdiff(colnames(x.train), "w0.dim"))] #move the extra column to the front
x.train <- as.matrix(x.train)
x.test <- mutate(x.test, "w0.dim" = 1)
x.test <- x.test[,c("w0.dim", setdiff(colnames(x.test), "w0.dim"))] #move the extra column to the front
x.test <- as.matrix(x.test)

sigmoid.func <- function(y.vec, x.mat, w){
  #exp(yi*xi%*%w)/(1+exp(yi*xi%*%w))
  #stable version:
  a <- y.vec*(x.mat %*% w)
  sapply(a, function(yxw) 1/(1+exp(yxw)))
}

#find w using steepest ascent algorithm
objective.func <- function(y.vec, x.mat, w){
  s <- sigmoid.func(y.vec, x.mat, w)
  s[s<10e-10] <- 10e-10
  sum(log(s))
}

gradient.loglikelihood <- function(y.vec, x.mat, w, sigmoids){
  #sigmoids <- sigmoid.func(y.vec, x.mat, w)
  gradient.matrix <- ((1-sigmoids)*y.vec) %*% x.mat
}

#initialize empty matrix to hold ws
w = matrix(NA, nrow = ncol(x.train), ncol = 10000) #Each column represents w at each iter
w[,1] = 0 
#w = vector(mode = "numeric", length = ncol(x.train)) 
#intialize empty vector to hold value of objective function
obj = vector(mode = "numeric", length = 10000)
obj[1] = objective.func(y.train, x.train, w[,1])
for(t in 1:9999){ 
  eta = 1/(10^5*sqrt(t+1))
  sigmoids <- sigmoid.func(y.train, x.train, w[,t])
  w[,t+1] = w[,t] + eta*gradient.loglikelihood(y.train, x.train, w[,t], sigmoids)
  obj[t+1] <- objective.func(y.train, x.train, w[,t+1])
}

summary(obj)
plot(obj[2:length(obj)], type = "l")

