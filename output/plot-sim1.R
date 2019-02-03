
for(num in 0:15){
  in.dir = paste0("/Users/ydd/Documents/covariance/output/sim1/batch",num)
  out.dir = "/Users/ydd/Documents/covariance/plots/"
  
  setwd(in.dir)
  filenames = list.files()
  tau_all = c()
  k_all = c()
  for(file in filenames){
    tau_all = c(tau_all,unlist(read.csv(file, header=F)))
    k_all = c(k_all,as.numeric(strsplit(unlist(strsplit(file,"k")),".csv")[[2]]))
  }
  tau_all = tau_all[tau_all!=0]
  tau_all = tau_all[tau_all!=300]
  #print(length(tau_all)/50)
  print(sum(k_all!=5))
  
  #pdf(paste0(out.dir, "hist", num,".pdf"), width= 5, height = 5)
  #hist(tau_all, nclass=60,xlim=c(0,300),ylim = c(0,50), main = '' , ylab = '', xlab = '',lwd=2)
  #abline(v = c(80,160,200), col = 'red',lty = 5, lwd=2)
  #dev.off()
}

