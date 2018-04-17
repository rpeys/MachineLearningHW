setwd("~/Documents/Columbia_DSI/machinelearning/HW/Homework 2/")

#### read in data ####
#train
x.train <- read.csv("hw2-data/X_train.csv", header = F)
y.train <- (read.csv("hw2-data/y_train.csv", header = F))$V1

##test
x.test <- read.csv("hw2-data/X_test.csv", header = F)
y.test <- read.csv("hw2-data/y_test.csv", header = F)$V1

#### question a - naive bayes ####
##establish naive bayes model
pi = mean(y.train)

theta.bern <- function(class, feature.ind){
  mean(x.train[y.train == class, feature.ind])
}

theta.pareto <- function(class, feature.ind){
  n = sum(y.train == class)
  n/sum(log(x.train[y.train == class, feature.ind]))
}

calc.bern.x <- function(class, x_vec, feature.ind){
  x <- x_vec[feature.ind]
  theta <- theta.bern(class, feature.ind)
  return(theta^x*(1-theta)^(1-x))
}

calc.pareto.x <- function(class, x_vec, feature.ind){
  x <- x_vec[feature.ind]
  theta <- theta.pareto(class, feature.ind)
  theta*x^(-(theta+1))
}

calc.prob.1.given.x <- function(x_vec){
  berns <- sapply(c(1:54), function(dim) calc.bern.x(1, x_vec, dim))
  paretos <- sapply(c(55:57), function(dim) calc.pareto.x(1, x_vec, dim))
  pi*prod(berns)*prod(paretos)
}

calc.prob.0.given.x <- function(x_vec){
  berns <- sapply(c(1:54), function(dim) calc.bern.x(0, x_vec, dim))
  paretos <- sapply(c(55:57), function(dim) calc.pareto.x(0, x_vec, dim))
  (1-pi)*prod(berns)*prod(paretos)
}

calc.prob.0.given.x(x.train[2,])
calc.prob.1.given.x(x.train[2,])

predict.naive.bayes <- function(x_vec){
  if(calc.prob.1.given.x(x_vec) > calc.prob.0.given.x(x_vec)) return(1) else return(0) 
}

predictions <- apply(x.test, 1, predict.naive.bayes)

sum(predictions == y.test)/length(y.test)

#for table in pdf:
sum(predictions == 0 & y.test == 0)
sum(predictions == 1 & y.test == 0)
sum(predictions == 0 & y.test == 1)
sum(predictions == 1 & y.test == 1)

confusionMatrix(predictions, y.test)

#### question 2b ####
library(tidyr)
library(plyr); library(dplyr)
library(ggplot2)

thetas.df <- data.frame("spam" = sapply(1:54, function(i) theta.bern(1, i)), "nonspam" = sapply(1:54, function(i) theta.bern(0, i)), "feature.index" = c(1:54))
thetas.df <- thetas.df %>% rowwise() %>% mutate("stem" = max(spam, nonspam))

ggplot(thetas.df) + geom_point(aes(x = feature.index, y=spam, color = "spam")) + geom_point(aes(x = feature.index, y=nonspam, color = "nonspam")) + geom_bar(aes(x=feature.index, y = stem), alpha = 0.3, stat = "identity", width = 0.1) + ggtitle("Thetas for ~Bern(feature|class) for Spam and Non-Spam Classes") + ylab("theta") + scale_x_continuous(breaks = c(1:54))

#look at difference in thetas at each feature
barplot(sapply(1:54, function(i) theta.bern(1, i)) - sapply(1:54, function(i) theta.bern(0, i)), names.arg = c(1:54))

#### question 2c ####
my.knn <- function(x_vec, k, x.train, y.train){
  differences <- t(apply(as.matrix(x.train), 1, function(row) row-x_vec))
  l1 <- apply(abs(differences), 1, sum)
  neighbors <- y.train[order(l1)[1:k]]
  if(sum(neighbors)/k >= 0.5) return(1) else return(0) #if tie, return spam
  #the below would choose ties truly randomly
  # if(sum(neighbors)/k > 0.5) { return(1) } else if(sum(neighbors)/k == 0.5){
  #   return(sample(c(0,1), 1)) #ties will be broken randomly
  # } else{
  #   return(0)
  # }
}

#record prediction accuracy for k of 1:20
pred.accuracy.k <- vector(length = 20, mode = "numeric")
for(k in 1:20){
  predictions <- apply(x.test, 1, function(row) my.knn(x_vec = unlist(row), k, x.train, y.train))
  pred.accuracy.k[k] <- sum(predictions==y.test)/length(predictions)
}
plot(pred.accuracy.k, type = "l", ylab = "prediction accuracy", xlab = "# of neighbors (k)", main = "Prediction Accuracy as a Function of K in KNN")
axis(1, at = c(1:20), labels = c(1:20)) # add more x axis tick marks

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
  sapply(a, function(yxw) 1/(1+exp(-yxw)))
}

#find w using steepest ascent algorithm
objective.func <- function(y.vec, x.mat, w){ #this is the log likelihood
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

plot(obj[2:length(obj)], type = "l", ylab = "value of objective function", main = "Logistic Regression Objective Function Per Iteration")

#### part e  - newtons ####

sigmoid.newton <- function(x.mat, w){
  #exp(yi*xi%*%w)/(1+exp(yi*xi%*%w))
  #stable version:
  a <- (x.mat %*% w)
  sapply(a, function(xw) 1/(1+exp(-xw)))
}

second.derv <- function(x.mat, w){
  sigs <- sigmoid.newton(x.mat, w)
  #print(length(sigs)) #4509
  sum = matrix(0, nrow = ncol(x.mat), ncol = ncol(x.mat)) #initialize sum
  for(i in 1:nrow(x.mat)){
    sum = sum + sigs[i]*(1-sigs[i])*(x.mat[i,] %*% t(x.mat[i,]))
  }
  return(-sum)
}

#initialize empty matrix to hold ws
library(MASS) #to take moore-penrose inverse ginv()
w = matrix(NA, nrow = ncol(x.train), ncol = 100) #Each column represents w at each iter
w[,1] = 0 
#w = vector(mode = "numeric", length = ncol(x.train)) 
#intialize empty vector to hold value of objective function
obj = vector(mode = "numeric", length = 100)
obj[1] = objective.func(y.train, x.train, w[,1])
for(t in 1:99){ 
  eta = 1/(sqrt(t+1))
  sigmoids <- sigmoid.func(y.train, x.train, w[,t])
  first.derv = gradient.loglikelihood(y.train, x.train, w[,t], sigmoids)
  sec.derv <- second.derv(x.train, w[,t])
  w[,t+1] = w[,t] - (eta * (ginv(sec.derv) %*% t(first.derv)))
  obj[t+1] <- objective.func(y.train, x.train, w[,t+1])
}

plot(obj, type = "l", ylab = "value of objective function", xlab = "iteration", main = "Logistic Regression Objective Function Per Iteration with Newton's method")

##predict on test set
#P(yi = +1|xi) = sigmoid(xi %*% w)
w = w[,100]
sigs <- sigmoid.newton(x.test, w)
plot(sigs)
preds <- ifelse(sigs > 0.5, 1, -1)
sum(preds==y.test)/length(y.test) #prediction accuracy

library(caret)
install.packages("e1071")
library(e1071)
confusionMatrix(preds, y.test)
