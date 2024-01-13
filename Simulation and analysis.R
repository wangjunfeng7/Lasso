library(boot)
library(MASS)
library(dplyr)
library(tidyverse)
library(bootstrap)
library(randomForestSRC)
library(reshape)
library(ggplot2)
library(e1071)
library(rpart)
library(rpart.plot)
library(Matrix)
library(xgboost)
library(rms)
library(glmnet)
library(logistf)
library(pROC)

###########################################################################################################################################
# Simulation Study
###########################################################################################################################################
##### Overall settings #####


set.seed(980313)
n_simulation <- 100   # Number of simulation
H  <- 30   # Number of predictors (10 true predictors, 20 false predictors)                 
N_population <- 3*10^6   # Total population (for correlation 0 and 0.2, 3*10^6, for correlation 0.4, 4*10^6)


beta <- c(0.2,0.2,0.4,0.4,0.7,0.7,0.9,0.9,1.1,1.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
event_rate <- c(0.01,0.02,0.05)
sample_size <- c(1000,5000,10000)
n_variables <- c(15,20,30)

n_true <- 10

validation_sample_size <- 10^6




##### Simulation starts here #####
# generate some data.frames to save results

n_performance_measure <- 11  # correlation,  strategy, event_rate, sample_size, n_variables,simulation, sens, spec, AUC, O:E, Brier







##### Data generation #####


cor<- c(0,0.2,0.4)


senario_correlation <- 1

mu <- as.vector(matrix(0, 1, H))   # mean 
sigma <- matrix(cor[senario_correlation],nrow=H,ncol=H)   # correlation matrix of predictors, 0/0.2/0.4
diag(sigma) <- rep(1,H)



##### generate multivariate distribution, 1 million  
data <- as.data.frame(mvrnorm(n=N_population, mu=mu, Sigma=sigma))      # Simulation per correlation matrix
names(data) <- c("X1", "X2","X3", "X4","X5", "X6","X7","X8","X9","X10",
                 "X11", "X12","X13", "X14","X15", "X16","X17","X18","X19","X20",
                 "X21", "X22","X23", "X24","X25", "X26","X27","X28","X29","X30")



# calculate lp and p
data$lp <- -4 + as.matrix(data) %*% beta  
data$p <- exp(data$lp)/(1+exp(data$lp))


# generate outcome
data$outcome <- rbinom(n = N_population, 1, data$p)


data_development <- data[c(1:(N_population*0.6)),]
data_validation <- data[c((N_population*0.6+1):N_population),]


### Save results

results<-data.frame(matrix(nrow = 0,ncol = n_performance_measure))


names(results) <- c("correlation",  "strategy", "event_rate", "sample_size", "n_variables","simulation", 
                    "sens", "spec", "AUC", "OE", "Brier")



for (e in 1:length(event_rate)) {
  
  data_val <- rbind(data_validation[data_validation$outcome==1,][c(1:(event_rate[e]*validation_sample_size)),],
                    data_validation[data_validation$outcome==0,][c(1:((1-event_rate[e])*validation_sample_size)),])
  
  
  
  for (i in 1:n_simulation){
    
    # create datasets
    
    data_dev_all <- rbind(data_development[data_development$outcome==1,][c((1+max(sample_size)*max(event_rate)*(i-1)):(max(sample_size)*max(event_rate)*i)),],
                          data_development[data_development$outcome==0,][c((1+max(sample_size)*(1-min(event_rate))*(i-1)):(max(sample_size)*(1-min(event_rate))*i)),])
    
    
    
    
    for (s in 1:length(sample_size)){
      
      data_dev <- rbind(data_dev_all[data_dev_all$outcome==1,][c(1:(event_rate[e]*sample_size[s])),],
                        data_dev_all[data_dev_all$outcome==0,][c(1:((1-event_rate[e])*sample_size[s])),])
      
      
      
      ##### fit models #####
      
      for (v in 1:length(n_variables)){
        
        ### 1. AUC_min
        
        model_AUC <- cv.glmnet(x=data.matrix(data_dev[,1:n_variables[v]]), y=data_dev$outcome, type.measure = "auc",alpha = 1, family="binomial")
        model_AUC_min <- glmnet(x=data.matrix(data_dev[,1:n_variables[v]]), y=data_dev$outcome,type.measure = "auc", alpha = 1, family = "binomial", lambda = model_AUC$lambda.min)
        
        
        ### 2. Class_min
        
        model_Class <- cv.glmnet(x=data.matrix(data_dev[,1:n_variables[v]]), y=data_dev$outcome, type.measure = "class",alpha = 1, family="binomial")
        model_Class_min <- glmnet(x=data.matrix(data_dev[,1:n_variables[v]]), y=data_dev$outcome,type.measure = "class", alpha = 1, family = "binomial", lambda = model_Class$lambda.min)
        
        
        ### 3. AUC_1se
        
        model_AUC_1se <- glmnet(x=data.matrix(data_dev[,1:n_variables[v]]), y=data_dev$outcome,type.measure = "auc", alpha = 1, family = "binomial", lambda = model_AUC$lambda.1se)
        
        
        ### 4. Class_1se
        
        model_Class_1se <- glmnet(x=data.matrix(data_dev[,1:n_variables[v]]), y=data_dev$outcome,type.measure = "class", alpha = 1, family = "binomial", lambda = model_Class$lambda.1se)
        
        
        ### 5. AUC_1se_refit
        ifelse(length(model_AUC_1se$beta[1:n_variables[v]][(model_AUC_1se$beta[1:n_variables[v]]>0) == TRUE])==0,
               model_AUC_1se_refit <-  glm(data = data_dev[,c(names(data_dev)[1],"outcome")], outcome~1,family = "binomial"),
               model_AUC_1se_refit <-  glm(data = data_dev[,c(names(data_dev)[(which(model_AUC_1se$beta[1:n_variables[v]]>0))],"outcome")], outcome~.,family = "binomial")
        )
        
        ### 6. Class_1se_refit
        
        ifelse(length(model_Class_1se$beta[1:n_variables[v]][(model_Class_1se$beta[1:n_variables[v]]>0) == TRUE])==0,
               model_Class_1se_refit <- glm(data = data_dev[,c(names(data_dev)[1],"outcome")], outcome~1,family = "binomial"),
               model_Class_1se_refit <- glm(data = data_dev[,c(names(data_dev)[(which(model_Class_1se$beta[1:n_variables[v]]>0))],"outcome")], outcome~.,family = "binomial")
        )
        
        
        
        ### 7. AUC_1se_Firth
        
        ifelse(length(model_AUC_1se$beta[1:n_variables[v]][(model_AUC_1se$beta[1:n_variables[v]]>0) == TRUE])==0,
               model_AUC_1se_Firth <- logistf(data = data_dev[,c(names(data_dev)[1],"outcome")], outcome~1),
               model_AUC_1se_Firth <- logistf(data = data_dev[,c(names(data_dev)[(which(model_AUC_1se$beta[1:n_variables[v]]>0))],"outcome")], outcome~.))
        
        
        ### 8. Class_1se_Firth
        ifelse(length(model_Class_1se$beta[1:n_variables[v]][(model_Class_1se$beta[1:n_variables[v]]>0) == TRUE])==0,
               model_Class_1se_Firth <- logistf(data = data_dev[,c(names(data_dev)[1],"outcome")], outcome~1),
               model_Class_1se_Firth <- logistf(data = data_dev[,c(names(data_dev)[(which(model_Class_1se$beta[1:n_variables[v]]>0))],"outcome")], outcome~.))
        
        

        
        
        
        ###### Results ######
        
        
        
        ### 1. AUC_min
        
        sens <- length(model_AUC_min$beta[1:10][(model_AUC_min$beta[1:10]>0) == TRUE])/n_true
        spec <- 1-length(model_AUC_min$beta[11:n_variables[v]][(model_AUC_min$beta[11:n_variables[v]]>0) == TRUE])/(n_variables[v]-n_true)
        
        data_val$pre_AUC_min<- predict(model_AUC_min, newx = data.matrix(data_val[,1:n_variables[v]]),type = "response")  # lasso model prediction 
        
        AUC <- auc(roc(data_val$outcome, data_val$pre_AUC_min))
        OE <- mean(data_val$outcome)/mean(data_val$pre_AUC_min)
        brier <- mean((data_val$outcome-data_val$pre_AUC_min)^2)
        
        
        temp_AUC_min <- c(cor[senario_correlation],"AUC_min",event_rate[e],sample_size[s],n_variables[v],i,
                          sens, spec, AUC, OE, brier )
        
        results <- rbind(results,temp_AUC_min)
        
        
        ### 2. Class_min
        
        sens <- length(model_Class_min$beta[1:10][(model_Class_min$beta[1:10]>0) == TRUE])/n_true
        spec <- 1-length(model_Class_min$beta[11:n_variables[v]][(model_Class_min$beta[11:n_variables[v]]>0) == TRUE])/(n_variables[v]-n_true)
        
        data_val$pre_Class_min<- predict(model_Class_min, newx = data.matrix(data_val[,1:n_variables[v]]),type = "response")  # lasso model prediction 
        
        AUC <- auc(roc(data_val$outcome, data_val$pre_Class_min))
        OE <- mean(data_val$outcome)/mean(data_val$pre_Class_min)
        brier <- mean((data_val$outcome-data_val$pre_Class_min)^2)
        
        
        temp_Class_min <- c(cor[senario_correlation],"Class_min",event_rate[e],sample_size[s],n_variables[v],i,
                            sens, spec, AUC, OE, brier )
        
        results <- rbind(results,temp_Class_min)
        
        
        ### 3. AUC_1se
        
        sens <- length(model_AUC_1se$beta[1:10][(model_AUC_1se$beta[1:10]>0) == TRUE])/n_true
        spec <- 1-length(model_AUC_1se$beta[11:n_variables[v]][(model_AUC_1se$beta[11:n_variables[v]]>0) == TRUE])/(n_variables[v]-n_true)
        
        data_val$pre_AUC_1se<- predict(model_AUC_1se, newx = data.matrix(data_val[,1:n_variables[v]]),type = "response")  # lasso model prediction 
        
        AUC <- auc(roc(data_val$outcome, data_val$pre_AUC_1se))
        OE <- mean(data_val$outcome)/mean(data_val$pre_AUC_1se)
        brier <- mean((data_val$outcome-data_val$pre_AUC_1se)^2)
        
        
        temp_AUC_1se <- c(cor[senario_correlation],"AUC_1se",event_rate[e],sample_size[s],n_variables[v],i,
                          sens, spec, AUC, OE, brier )
        
        results <- rbind(results,temp_AUC_1se)
        
        ### 4. Class_1se
        
        sens <- length(model_Class_1se$beta[1:10][(model_Class_1se$beta[1:10]>0) == TRUE])/n_true
        spec <- 1-length(model_Class_1se$beta[11:n_variables[v]][(model_Class_1se$beta[11:n_variables[v]]>0) == TRUE])/(n_variables[v]-n_true)
        
        data_val$pre_Class_1se<- predict(model_Class_1se, newx = data.matrix(data_val[,1:n_variables[v]]),type = "response")  # lasso model prediction 
        
        AUC <- auc(roc(data_val$outcome, data_val$pre_Class_1se))
        OE <- mean(data_val$outcome)/mean(data_val$pre_Class_1se)
        brier <- mean((data_val$outcome-data_val$pre_Class_1se)^2)
        
        
        temp_Class_1se <- c(cor[senario_correlation],"Class_1se",event_rate[e],sample_size[s],n_variables[v],i,
                            sens, spec, AUC, OE, brier )
        
        results <- rbind(results,temp_Class_1se)
        
        ### 5. AUC_1se_refit
        
        sens <- length(model_AUC_1se$beta[1:10][(model_AUC_1se$beta[1:10]>0) == TRUE])/n_true
        spec <- 1-length(model_AUC_1se$beta[11:n_variables[v]][(model_AUC_1se$beta[11:n_variables[v]]>0) == TRUE])/(n_variables[v]-n_true)
        
        data_val$pre_AUC_1se_refit<- predict(model_AUC_1se_refit, data_val[,c(names(data_dev)[(which(model_AUC_1se$beta[1:n_variables[v]]>0))],"outcome","X1")],type = "response")  # lasso model prediction 
        
        AUC <- auc(roc(data_val$outcome, data_val$pre_AUC_1se_refit))
        OE <- mean(data_val$outcome)/mean(data_val$pre_AUC_1se_refit)
        brier <- mean((data_val$outcome-data_val$pre_AUC_1se_refit)^2)
        
        
        temp_AUC_1se_refit <- c(cor[senario_correlation],"AUC_1se_refit",event_rate[e],sample_size[s],n_variables[v],i,
                                sens, spec, AUC, OE, brier )
        
        results <- rbind(results,temp_AUC_1se_refit)
        
        ### 6. Class_1se_refit
        
        sens <- length(model_AUC_1se$beta[1:10][(model_Class_1se$beta[1:10]>0) == TRUE])/n_true
        spec <- 1-length(model_AUC_1se$beta[11:n_variables[v]][(model_Class_1se$beta[11:n_variables[v]]>0) == TRUE])/(n_variables[v]-n_true)
        
        data_val$pre_Class_1se_refit<- predict(model_Class_1se_refit, data_val[,c(names(data_dev)[(which(model_Class_1se$beta[1:n_variables[v]]>0))],"outcome","X1")],type = "response")  # lasso model prediction 
        
        AUC <- auc(roc(data_val$outcome, data_val$pre_Class_1se_refit))
        OE <- mean(data_val$outcome)/mean(data_val$pre_Class_1se_refit)
        brier <- mean((data_val$outcome-data_val$pre_Class_1se_refit)^2)
        
        
        temp_Class_1se_refit <- c(cor[senario_correlation],"Class_1se_refit",event_rate[e],sample_size[s],n_variables[v],i,
                                  sens, spec, AUC, OE, brier )
        
        results <- rbind(results,temp_Class_1se_refit)
        
        
        ### 7. AUC_1se_Firth
        
        sens <- length(model_AUC_1se$beta[1:10][(model_AUC_1se$beta[1:10]>0) == TRUE])/n_true
        spec <- 1-length(model_AUC_1se$beta[11:n_variables[v]][(model_AUC_1se$beta[11:n_variables[v]]>0) == TRUE])/(n_variables[v]-n_true)
        
        data_val$pre_AUC_1se_Firth<- predict(model_AUC_1se_Firth, data_val[,c(names(data_dev)[(which(model_AUC_1se$beta[1:n_variables[v]]>0))],"outcome","X1")],type = "response")  # lasso model prediction 
        
        AUC <- auc(roc(data_val$outcome, data_val$pre_AUC_1se_Firth))
        OE <- mean(data_val$outcome)/mean(data_val$pre_AUC_1se_Firth)
        brier <- mean((data_val$outcome-data_val$pre_AUC_1se_Firth)^2)
        
        
        temp_AUC_1se_Firth <- c(cor[senario_correlation],"AUC_1se_Firth",event_rate[e],sample_size[s],n_variables[v],i,
                                sens, spec, AUC, OE, brier )
        
        results <- rbind(results,temp_AUC_1se_Firth)
        
        
        ### 8. Class_1se_Firth
        
        sens <- length(model_AUC_1se$beta[1:10][(model_Class_1se$beta[1:10]>0) == TRUE])/n_true
        spec <- 1-length(model_AUC_1se$beta[11:n_variables[v]][(model_Class_1se$beta[11:n_variables[v]]>0) == TRUE])/(n_variables[v]-n_true)
        
        data_val$pre_Class_1se_Firth<- predict(model_Class_1se_Firth, data_val[,c(names(data_dev)[(which(model_Class_1se$beta[1:n_variables[v]]>0))],"outcome","X1")],type = "response")  # lasso model prediction 
        
        AUC <- auc(roc(data_val$outcome, data_val$pre_Class_1se_Firth))
        OE <- mean(data_val$outcome)/mean(data_val$pre_Class_1se_Firth)
        brier <- mean((data_val$outcome-data_val$pre_Class_1se_Firth)^2)
        
        
        temp_Class_1se_Firth <- c(cor[senario_correlation],"Class_1se_Firth",event_rate[e],sample_size[s],n_variables[v],i,
                                  sens, spec, AUC, OE, brier )
        
        results <- rbind(results,temp_Class_1se_Firth)
        
        
      }}}}


names(results) <- c("correlation",  "strategy", "event_rate", "sample_size", "n_variables","simulation", 
                    "sens", "spec", "AUC", "OE", "Brier")