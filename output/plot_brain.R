require(ggplot2)
library('reshape')
library(png)

#Replace the directory and file information with your info
img <- readPNG("/Users/ydd/Documents/covariance/background_image/brain2.png")


for (id in 1:length(Sigma_t)){
  df_cor = as.data.frame(cov2cor(Sigma_t[[id]]))
  colnames(df_cor) = df_mapping$names
  
  df_cor$roi = df_mapping$names
  
  df_cor_metled = melt(df_cor, id=c("roi"))
  df_cor_off = df_cor_metled[df_cor_metled$roi != df_cor_metled$variable,]
  df_cor_off = df_cor_off[as.character(df_cor_off$roi) < as.character(df_cor_off$variable),]
  df_connected = df_cor_off[abs(df_cor_off$value)>0.5,]
  
  df_network = data.frame()
  for(j in 1:dim(df_connected)[1]){
    df_temp = rbind(df_mapping[df_mapping$names==df_connected[j,1],],df_mapping[df_mapping$names==df_connected[j,2],])
    df_temp$grp = j
    df_network = rbind(df_network, df_temp)
  }
  df1 = df_network
  df1$names = NA
  df2 = df_mapping
  df2$grp = nrow(df1)/2+1:nrow(df2)
  df_plot = rbind(df1, df2)
  
  p = ggplot(df_plot, aes(x,y,group=grp)) + 
    annotation_custom(rasterGrob(img, width = unit(1,"npc"), height = unit(1,"npc")), -Inf, Inf, -Inf, Inf) + geom_text(aes(label=names), na.rm = TRUE) + geom_line() + xlim(-49, -15) + ylim(-58, -20) +
    xlab('') + ylab('') + 
    theme(axis.text.x=element_blank(), axis.ticks.x=element_blank())  +
    theme(axis.text.y=element_blank(), axis.ticks.y=element_blank()) 
  pdf(paste0("BrainImage",id,".pdf"), width= 6, height = 6)
  print(p)
  dev.off()
}
