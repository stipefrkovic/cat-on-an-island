setwd("/home/sf/Downloads/University/RLP/rlp-assignments/finalproject/results")
rlp_data <- read.csv("results.csv", header=TRUE, stringsAsFactors=FALSE)
aggregate(rlp_data$time_steps,list(rlp_data$algorithm), function(x) c(mean = mean(x), sd = sd(x), var = var(x)))
anova <- aov(time_steps ~ algorithm, data = rlp_data)
summary(anova)
tukey = TukeyHSD(anova)
tukey
plot(tukey)