# library(devtools)
# Sys.setenv("R_REMOTES_NO_ERRORS_FROM_WARNINGS" = "true")
# devtools::install_github("lotze/COMPoissonReg", ref = "v0.7.1")
library(COMPoissonReg)

# wd <- dirname(rstudioapi::getSourceEditorContext()$path)
wd <- "D:/GitHub/COM_POISSON/runRcode"

d <- read.csv(paste0(wd, '/fitData.csv'), header = F)
# newD <- read.csv(paste0(wd, '/basData.csv'), header = F)
# theta <- read.csv(paste0(wd, '/theta.csv'), header = F)


y <- d[, 1]
X <- d[, 2:ncol(d)]
fitData <- data.frame(y, X)
# names(newD) <- names(fitData[, 2:ncol(fitData)])

cmpFit <- glm.cmp(y ~ .,
                  y ~ ., data = fitData)
beta <- cmpFit$beta
gam <- cmpFit$gamma

write.csv(data.frame(beta, gam),
          file = paste0(wd, '/cmp_t1.csv'))

# nu(cmpFit)
# cmpFitted<- predict(cmpFit, newdata=newD)



# posFit <- glm(y ~ ., data = fitData,
#               family=poisson)
# posFitted <- predict(posFit, newdata=newD, type = 'response')
# 
# x0 <- seq(0, 2*pi, length.out = 256)
# 
# plot(theta$V1, y)
# lines(x0, cmpFitted, col = 'red')
# lines(x0, posFitted, col = 'blue')






