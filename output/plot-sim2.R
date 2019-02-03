################################################
## heatmap plot (covariance and factor loadings)
################################################

library(reshape2)
library(ggplot2)

in.dir = "/Users/ydd/Documents/covariance/output/simulation/"
out.dir = "/Users/ydd/Documents/covariance/plots/"
filename = 'Beta_delta20'
Beta = read.csv(paste0(in.dir, filename,'.csv'), header=F)
Beta$id = 1:dim(Beta)[1]

melted_beta <- melt(abs(Beta), id = 'id')

pdf(paste0(out.dir, filename,".pdf"), width= 6, height = 12)
p <- ggplot(data = melted_beta, aes(x=variable, y=id, fill=value)) + theme_bw() + geom_tile() + scale_fill_gradient(low="white", high="royalblue") + xlab('') + ylab('') + scale_x_discrete(expand = c(0, 0)) + scale_y_reverse(expand = c(0, 0)) + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank()) + theme(axis.text.x=element_blank(), axis.ticks.x=element_blank()) + theme(legend.position="none")
p
dev.off()


in.dir = "/Users/ydd/Documents/covariance/output/simulation/"
out.dir = "/Users/ydd/Documents/covariance/plots/"
filename = 'cov_true'
covmat = read.csv(paste0(in.dir, filename,'.csv'), header=F)
covmat$id = 1:dim(covmat)[1]

melted_covmat <- melt(abs(covmat), id = 'id')
pdf(paste0(out.dir, filename,".pdf"), width= 12, height = 12)
p <- ggplot(data = melted_covmat, aes(x=variable, y=id, fill=value)) + theme_bw() + geom_tile() + scale_fill_gradient(low="white", high="royalblue") + xlab('') + ylab('') + scale_x_discrete(expand = c(0, 0)) + scale_y_reverse(expand = c(0, 0)) + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank()) + theme(axis.text.x=element_blank(), axis.ticks.x=element_blank()) + theme(legend.position="none")
p
dev.off()


#filename = 'Lambda2_delta10'
#Lambda2 = read.csv(paste0(in.dir, filename,'.csv'), header=F)
