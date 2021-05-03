library(COMPoissonReg)
wd <- "D:/GitHub/COM_POISSON/demo/hc"

######
y <- read.csv(paste0(wd, '/y.csv'), header = F)
X <- read.csv(paste0(wd, '/X.csv'), header = F)
G <- read.csv(paste0(wd, '/G.csv'), header = F)

fitData <- data.frame(y = y, X = X, G = G)
xnam <- paste("x", 0:(ncol(X)-1), sep="")
gnam <- paste("g", 0:(ncol(G)-1), sep="")

names(fitData)[1] <- 'y' 
names(fitData)[2:(ncol(X) + 1)] <- xnam
names(fitData)[(ncol(X) + 2):ncol(fitData)] <- gnam

if(length(xnam) > 1){xnam2 <- xnam[-1]}else{xnam2 <- 1}
if(length(gnam) > 1){gnam2 <- gnam[-1]}else{gnam2 <- 1}

fmlaX <- as.formula(paste('y~',
                          paste(xnam2, collapse= "+")))
fmlaG <- as.formula(paste('y~',
                          paste(gnam2, collapse= "+")))

cmpFit <- glm.cmp(fmlaX,
                  fmlaG, data = fitData)
beta <- unname(cmpFit$beta)
gam <- unname(cmpFit$gamma)

write.csv(c(beta, gam),
          file = paste0(wd, '/cmp_t1.csv'), row.names = F)


