library(reshape)
library(ggplot2)

retMat=read.csv('../data/sp100retMat.csv')
rownames(retMat) = retMat$X
retMat = retMat[,-1]

in.dir = './sp100/'
out.dir = "../plots/"
filename = 'Beta'
Beta = read.csv(paste0(in.dir, filename,'.csv'), header=F)
colnames(Beta) = 1:20
rownames(Beta) = colnames(retMat)
#Beta$id = rownames(Beta)

melted_beta <- melt(as.matrix(abs(Beta)))
colnames(melted_beta) = c('Ticker', 'Factor', 'value')

pdf(paste0(out.dir, "Beta_sp.pdf"), width= 12, height = 4)
p <- ggplot(data = melted_beta, aes(x=Ticker, y=Factor, fill=value)) + 
  theme_bw() + geom_tile() + scale_fill_gradient(low="white", high="royalblue") + 
  xlab('') + ylab('') + scale_x_discrete(expand = c(0, 0), position = "top") + 
  scale_y_reverse(expand = c(0, 0)) + 
  theme(axis.text.y=element_blank(), axis.ticks.y=element_blank()) + 
  theme(text = element_text(size=10),axis.text.x.top=element_text(angle=45, hjust=0.2)) + 
  theme(legend.position="none")
p
dev.off()

for(j in 2:10){
  print(j)
  print(rownames(Beta)[abs(Beta[,j])>0.2])
}

filename = 'tau'
tau = unlist(read.csv(paste0(in.dir, filename,'.csv'), header=F))
Date = rownames(retMat)
Date[tau]
