setwd("~/Documents/Columbia_DSI/machinelearning/HW/homework1/")

x.train <- as.matrix(read.csv("hw1-data/X_train.csv", header = F))
y.train <- as.matrix(read.csv("hw1-data/y_train.csv", header = F))

#loss.func <- lambda*sum(weights^2) + sum(sapply(1:350, function(i) sum((y[i,] - x[i,]*weights)^2)))

##function to solve for weights in ridge regression
W_ridgereg <- function(lambda, X, Y){
  solve(lambda * diag(ncol(X)) + t(X) %*% X) %*% t(X) %*% Y
}
  
##function to solve for the numbers of degrees of freedom for each lambda
deg.free <- function(lambda, X){
  singular.x <- svd(X)$d
  sum(singular.x^2 / (lambda + singular.x^2))
}

##initialize vectors/matrices to hold results
w_mat <- matrix(nrow = 7, ncol = 5001) #each column corresponds to a different value of lambda, each row to a different feature
df_vec <- vector(mode = "numeric", length = 5001)
for(lambda in 0:5000){
  w_mat[,lambda+1] <- W_ridgereg(lambda, x.train, y.train) #the values of lambda are one off from the row index (eg. lambda=0 is in row 1)
  df_vec[lambda+1] <- deg.free(lambda, x.train) #the values of lambda are one off from the row index (eg. lambda=0 is in index 1)
}

##plot
colors <- c("black", "red", "orange", "pink", "green", "blue", "purple")
plot(df_vec, w_mat[1,], ylim = c(floor(min(w_mat)), ceiling(max(w_mat))), cex=0.5, col = colors[1], xlab = "df(lambda)", ylab = "weight (ridge regression)", main = "Feature Weights as a Function of Degrees of Freedom of Lambda")
points(df_vec, w_mat[2,], cex=0.5, col = colors[2])
points(df_vec, w_mat[3,], cex=0.5, col = colors[3])
points(df_vec, w_mat[4,], cex=0.5, col = colors[4])
points(df_vec, w_mat[5,], cex=0.5, col = colors[5])
points(df_vec, w_mat[6,], cex=0.5, col = colors[6])
points(df_vec, w_mat[7,], cex=0.5, col = colors[7])
legend("bottomleft", legend = paste("x feature", c(1:7)), fill = colors)

#### part 1C - predict ####
x.test <- as.matrix(read.csv("hw1-data/X_test.csv", header = F))
y.test <- as.matrix(read.csv("hw1-data/y_test.csv", header = F))
predictions <- x.test %*% w_mat[,1:51]
rmse = sqrt(1/42*apply((y.test[,1] - predictions), 2, function(col) sum(col^2))) #each column shows
plot(0:50, rmse, main = "Mean Squared Error as a Function of Lambda", ylab = "RMSE", xlab = "lambda")

#### part 2 ####
##p=e
predictions <- x.test %*% w_mat[,1:501]
rmse = sqrt(1/42*apply((y.test[,1] - predictions), 2, function(col) sum(col^2))) #each column shows
##p=2
#train
x.train.poly2 <- cbind(x.train[,7], x.train[,1:6], (x.train[,1:6])^2)
w_mat.poly2 <- matrix(nrow = ncol(x.train.poly2), ncol = 501) #each row corresponds to a different value of lambda, each column to a different feature
for(lambda in 0:500){
  w_mat.poly2[,lambda+1] <- W_ridgereg(lambda, x.train.poly2, y.train) #the values of lambda are one off from the row index (eg. lambda=0 is in row 1)
}
#test
x.test.poly2 <- cbind(x.test[,7], x.test[,1:6], (x.test[,1:6])^2)
predictions <- x.test.poly2 %*% w_mat.poly2
rmse.poly2 = sqrt(1/42*apply((y.test[,1] - predictions), 2, function(col) sum(col^2))) #each column shows
plot(0:500, rmse.poly2, main = "Mean Squared Error as a Function of Lambda", ylab = "RMSE", xlab = "lambda")

##p=3
#train
x.train.poly3 <- cbind(x.train[,7], x.train[,1:6], (x.train[,1:6])^2, (x.train[,1:6])^3)
w_mat.poly3 <- matrix(nrow = ncol(x.train.poly3), ncol = 501) #each row corresponds to a different value of lambda, each column to a different feature
for(lambda in 0:500){
  w_mat.poly3[,lambda+1] <- W_ridgereg(lambda, x.train.poly3, y.train) #the values of lambda are one off from the row index (eg. lambda=0 is in row 1)
}
#test
x.test.poly3 <- cbind(x.test[,7], x.test[,1:6], (x.test[,1:6])^2, (x.test[,1:6])^3)
predictions <- x.test.poly3 %*% w_mat.poly3
rmse.poly3 = sqrt(1/42*apply((y.test[,1] - predictions), 2, function(col) sum(col^2))) #each column shows
plot(0:500, rmse.poly3, main = "Mean Squared Error as a Function of Lambda", ylab = "RMSE", xlab = "lambda")

##plot all 3
plot(0:500, rmse, main = "Mean Squared Error as a Function of Lambda\nfor Polynomial Regressions of Orders 1, 2, and 3", ylab = "RMSE", xlab = "lambda", cex = 0.5, ylim = c(floor(min(rmse, rmse.poly2, rmse.poly3)), ceiling(max(c(rmse, rmse.poly2, rmse.poly3)))))
points(0:500, rmse.poly2, main = "Mean Squared Error as a Function of Lambda", ylab = "RMSE", xlab = "lambda", cex = 0.5, col=colors[2])
points(0:500, rmse.poly3, main = "Mean Squared Error as a Function of Lambda", ylab = "RMSE", xlab = "lambda", cex = 0.5, col = colors[3])
legend("bottomright", legend = c("1st order polynomial", "2nd order polynomial", "3rd order polynomial"), fill = colors[1:3])

