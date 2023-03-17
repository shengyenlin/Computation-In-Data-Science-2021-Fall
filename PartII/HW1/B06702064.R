#上課內容
census <- read.csv(file = 'HW1_census-tract_data.csv', header = TRUE, fill = FALSE, row.names = 1)
Z <- apply(census, 2, scale)
X_bar = round(apply(census, 2, mean), 2)
X_bar
S_origin <- round(var(census), 3)
S_origin
eigen_original <- eigen(S_origin) #do PCA on covariance matrix
eigen_original

PC <- t(eigen_original$vectors) %*% t(data.matrix(census))
corr <- cor(t(PC), census)
t(corr)

#(a)
census <- read.csv(file = 'HW1_census-tract_data.csv', header = TRUE, fill = FALSE, row.names = 1)
census_100 <- census
census_100$Median.value.home..10.000s. <- census_100$Median.value.home..10.000s. * 100
Z_100 <- apply(census_100, 2, scale)
Z_100

S_100 <- round(var(census_100), 3)
S_100

eigen_100 <- eigen(S_100)
eigen_100$values
round(eigen_100$vectors, 3)

#(b)
eigen_100$values[1] / sum(eigen_100$values)
eigen_100$values[2] / sum(eigen_100$values)

#(c)
PC_100 <- t(eigen_100$vectors) %*% t(data.matrix(census_100))
corr_100 <- cor(t(PC_100), Z_100)
t(corr_100)
