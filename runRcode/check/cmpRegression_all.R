library(COMPoissonReg)

wd <- dirname(rstudioapi::getSourceEditorContext()$path)
Y <- read.csv(paste0(wd, '/ry.csv'), header = F)
X <- read.csv(paste0(wd, '/X.csv'), header = F)
newD <- read.csv(paste0(wd, '/basData.csv'), header = F)
theta <- read.csv(paste0(wd, '/theta.csv'), header = F)

x0 <- seq(0, 2*pi, length.out = 256)

CMP_mean <- matrix(NA, nrow = 256, ncol = ncol(Y))
CMP_lam <- matrix(NA, nrow = 256, ncol = ncol(Y))
CMP_nu <- matrix(NA, nrow = 256, ncol = ncol(Y))
CMP_beta <- matrix(NA, nrow = ncol(Y), ncol = ncol(X) + 1)
CMP_gamma <- matrix(NA, nrow = ncol(Y), ncol = ncol(X) + 1)


for(i in 1:ncol(Y)){
  
  y <- Y[, i]
  fitData <- data.frame(y, X)
  
  cmpFit <- glm.cmp(y ~ .,
                    y ~ ., data = fitData)
  
  
  CMP_beta[i, ] <- cmpFit$beta
  CMP_gamma[i, ] <- cmpFit$gamma
  
  CMP_mean[, i]<- predict(cmpFit, newdata=newD)
  
  CMP_lam[, i] <- exp(as.matrix(cbind(1, newD)) %*% cmpFit$beta)
  CMP_nu[, i] <- exp(as.matrix(cbind(1, newD)) %*% cmpFit$gamma)
  
}

# save.image(paste0(wd, '/neuron72_v2.RData'))

##############
#### progression of parameters
plot(CMP_beta[, 1], type = 'l')
plot(CMP_beta[, 2], type = 'l')
plot(CMP_beta[, 3], type = 'l')
plot(CMP_beta[, 4], type = 'l')
plot(CMP_beta[, 5], type = 'l')
plot(CMP_beta[, 6], type = 'l')
plot(CMP_beta[, 7], type = 'l')
plot(CMP_beta[, 8], type = 'l')



plot(CMP_gamma[, 1], type = 'l')
plot(CMP_gamma[, 2], type = 'l')
plot(CMP_gamma[, 3], type = 'l')
plot(CMP_gamma[, 4], type = 'l')
plot(CMP_gamma[, 5], type = 'l')
plot(CMP_gamma[, 6], type = 'l')
plot(CMP_gamma[, 7], type = 'l')
plot(CMP_gamma[, 8], type = 'l')

##############
#### progression of tuning curves
myvcmp <- function(theta, maxSum = 200){
  lam <- theta[1]
  nu <- theta[2]
  return(sum((0:maxSum)^2*dcmp((0:maxSum), lam, nu))-
           (ecmp(lam, nu))^2)
}

FF <- matrix(NA, nrow = 256, ncol = ncol(Y))

for(i in 1:ncol(Y)){
  
  ff <- apply(data.frame(CMP_lam[, i], CMP_nu[, i]),1, myvcmp)/
    CMP_mean[, i]
  FF[, i] <- ff
  # png(sprintf(paste0(wd, '/plots/img%03d.png'),i),
  #     width = 600,height = 600)
  # plot(theta$V1, Y[, i], xlab = 'direction', ylab = 'y',
  #      main = paste('trial =', i), ylim = range(Y))
  # lines(x0, CMP_mean[, i], col = 'red', lwd = 2)
  # 
  # lines(x0, ff, col = 'blue', lwd = 2)
  # abline(h = 1)
  # legend('topright', legend = c('mean', 'fano'),
  #        lwd = 2, col = c('red', 'blue'))
  # dev.off()
  
}

plot(FF[1, ], type = 'l', col = 1, ylim = range(FF))
for(j in seq(2, 256, 1)){
  lines(FF[j, ], type = 'l', col = j)
}

plot(CMP_mean[1, ], type = 'l', col = 1, ylim = range(CMP_mean))
for(j in 1:256){
  lines(CMP_mean[j, ], type = 'l', col = j)
}













