rm(list=ls())
library(graphon)
library(dplyr)
library(ggplot2)
source("network_generate.R")

library(pheatmap)
library(viridis)
library(parallel)
library(gridExtra)
library(grid)
library(graphon)

source("auxiliary.R")
source("function.R")
bk = seq(0, 1, 0.01)

library(igraph)
library(transport)
library(gtools)
library(reticulate)



# Ensure Python dependencies
pot = import("ot")
np = import("numpy")

## epsilon = 1e-5

sample_size_t = 50

graphon_id_list_s  =  1:10

sample_size_s_list = seq(100, 1000, 100)


nrepeat = 50
# nrepeat = 10
cores = 10
results_all = NULL
type = "unif"
for(graphon_id_s in graphon_id_list_s){
  
  for(graphon_id_t in graphon_id_s){
    # for(graphon_id_t in graphon_id_list_t){
    for(sample_size_s in sample_size_s_list){
      
      print(c(graphon_id_s,graphon_id_t,sample_size_s))
      repeat_results = mclapply(1:nrepeat,function(iii){
        set.seed(iii)
        gp = gp_generate(sample_size_t, graphon_id_t,type=type)
        P_t = gp$P
        # M = matrix(runif(sample_size_t * sample_size_t,
        #                  min = -0.05, max = 0.05), nrow = sample_size_t, ncol = sample_size_t)
        # P_t = P_t+M
        P_t = (P_t + t(P_t))/2 
        diag(P_t)=0
        P_t[P_t>1]= 1
        P_t[P_t<0]= 0
        A_t = gmodel.P(P_t,rep=1,symmetric.out=TRUE)
        diag(A_t) = 0
        
        gp = gp_generate(sample_size_s, graphon_id_s,type=type)
        P = gp$P 
        # Generate a random matrix of size n x n with values drawn from uniform(-0.005, 0.005)
        M = matrix(runif(sample_size_s * sample_size_s,
                         min = -0.01, max = 0.01), nrow = sample_size_s, ncol = sample_size_s)
        P = P+M
        P = (P + t(P))/2 
        diag(P)=0
        P[P>1]= 1
        P[P<0]= 0
        A_s = gmodel.P(P,rep=1,symmetric.out=TRUE)
        diag(A_s) = 0
        p_source = P  
        
        ###USVT for target only####
        p_t_usvt = est.USVT(A_t)$P
        diag(p_t_usvt) = 0
        error_USVT = mean((P_t -p_t_usvt )^2)
        
        ###ICE for target only####
        source("ICE.R")
        ice_res = ICE(A = A_t, C_it = 0.3, C_est = 1, P_hat_0 = NULL, rounds = 10)
        #  C_est = 0.7
        p_t_ICE = ice_res$final_P_hat
        diag(p_t_ICE)=0
        error_ICE = mean((P_t - p_t_ICE )^2)
        
        
        ###SAS for target only####
        p_t_lg = est.LG(A_t)$P
        diag(p_t_lg) = 0
        error_SAS = mean((P_t - p_t_lg )^2)
        
        ####Transfer learning##########
        ### result = Graphon_Trans(A_s, A_t,epsilon = 1e-5)
        # ---- epsilon = 0.01 ----
        result_1 = Graphon_Trans(A_s, A_t, epsilon = 0.01)
        
        error_debias_1 = mean((result_1$phat_final  - P_t)^2)
        error_trans_1  = mean((result_1$phat_t_trasport - P_t)^2)
        error_NS       = mean((result_1$phat_ns - P_t)^2)
        error_notrans_DS = mean((result_1$phat_notrans_DS - P_t)^2)
        gw_distance    = result_1$gw_distance
       
        cat("[🔍] GW Distance for EGW: ", gw_distance, "\n")
        
        # ---- epsilon = 1e-5 ----
        result_2 = Graphon_Trans(A_s, A_t, epsilon = 1e-5)
        
        error_debias_2 = mean((result_2$phat_final  - P_t)^2)
        error_trans_2  = mean((result_2$phat_t_trasport - P_t)^2)
        
        # ---- Combine into result ----
        result = data.frame(
          gw_distance = gw_distance,
          NS = error_NS, 
          ICE = error_ICE,
          USVT = error_USVT,
          SAS = error_SAS, 
          Notrans_DS = error_notrans_DS,
          GTrans_0.01 = error_debias_1,
          GTrans2_0.01 = error_trans_1,
          GTrans_1e.5 = error_debias_2,
          GTrans2_1e.5 = error_trans_2
        )
        
        return(result)
        
      },mc.cores = cores)
      
      results = do.call("rbind",repeat_results)
      results$graphon_id_s = graphon_id_s
      results$graphon_id_t = graphon_id_t
      results$sample_size_s = sample_size_s
      print(results)
      results_all = rbind(results_all,results)
      save(results_all,file="results_all_increasingN.rda")
    }
    
    library(tidyr)
    library(patchwork)
    plot_list = list()  # To store individual plots
    
    for(i_s in 1:length(graphon_id_list_s)){
      # for(i_t in graphon_id_list_t){
      
      results_all_df = as.data.frame(results_all)
      temp = filter(results_all_df,graphon_id_s %in% graphon_id_list_s[i_s],
                    graphon_id_t %in% graphon_id_list_s[i_s] )
      # Load necessary libraries
      library(ggplot2)
      library(dplyr)
      library(tidyr)
      
      gw_distance = temp$gw_distance
      temp = temp[,-which("gw_distance" %in% colnames(temp))]
      
      # Reshape the data to long format for easier plotting
      data_long = pivot_longer(
        temp,
        ## cols = c(NS, ICE, USVT, SAS, GTrans,GTrans2,Notrans_DS),  # Error columns to compare
        cols = c("NS", "ICE", "USVT", "SAS", 
                 "GTrans_0.01", "GTrans2_0.01", 
                 "GTrans_1e.5", "GTrans2_1e.5", 
                 "Notrans_DS"),
        names_to = "Method",                  # Name of the new column for methods
        values_to = "Error"                   # Name of the new column for error values
      )
      
      #
      averaged_data = data_long %>%
        group_by(sample_size_s, Method) %>%
        summarize(
          Average_Error = mean(Error, na.rm = TRUE),
          SE = sd(Error, na.rm = TRUE),
          .groups = "drop"
        )
      
      
      # Create the line chart with averaged errors
      # averaged_data$Method = factor(averaged_data$Method, 
      #                               levels=c("GTrans","GTrans2","NS","USVT","ICE","SAS","Notrans_DS"))
      # # averaged_data = filter(averaged_data,! Method %in% "Notrans_DS")
      averaged_data$Method = factor(averaged_data$Method, 
                                    levels = c("GTrans_0.01", "GTrans2_0.01", 
                                               "GTrans_1e.5", "GTrans2_1e.5", 
                                               "NS", "USVT", "ICE", "SAS", "Notrans_DS"))
      
      plot = ggplot(averaged_data, aes(x = sample_size_s, y = Average_Error, 
                                       color = Method, 
                                       shape=Method,
                                       linetype= Method,
                                       group = Method))  +
        geom_line(size = 1.2) +               # Line connecting averaged points
        geom_point(size = 3, alpha = 0.8) +  # Add points for averaged errors
        # geom_errorbar(aes(ymin = Average_Error - SE, ymax = Average_Error + SE),
        #               width = 0.01, size = 0.8) +
        theme_bw() +
        labs( 
          # title = paste0(i_s,",",i_t),
          x = "Sample Size (s)",y = "Average Error") +  
        theme(legend.position = "none",
              legend.title = element_blank(),    # Remove legend title
              # panel.grid.minor = element_blank()
        ) + 
        # scale_shape_manual(values = c(16, 15, 17, 18,0,1,2)) +
        # scale_color_manual(values = c("#C75736", "purple","#3A6DA8","#F2C201","#7A9832","grey","black")) 
        # scale_shape_manual(values = c(16, 15, 17, 18, 0, 1, 2, 5, 6)) +
        # Add 2 new shapes and colors at the beginning (for the new GTrans variants)
        scale_shape_manual(values = c(8, 9, 16, 15, 17, 18, 0, 1, 2)) + 
        scale_color_manual(values = c("orange", "#33A02C", "#C75736", "purple", 
                                      "#3A6DA8", "#F2C201", "#7A9832", "grey", "black"))
      
      
      plot_list = append(plot_list, list(plot))
      
      # }
    }
    library(gridExtra)
    # Save the grid layout manually using png()
    png("graphon_error_N.png", width = 4000, height = 1500, res = 200)
    grid.arrange(grobs = plot_list, ncol = ceiling(length(graphon_id_list_s)/2), nrow = 2)
    dev.off()
    
    
  }
}

# 
load("results_all_increasingN.rda")
results_all$gw_distance2 = results_all$gw_distance/results_all$sample_size_s

library(colorspace)

deep_pink <- darken("pink", amount = 0.2)  


hist(results_all$gw_distance)
head(results_all)
sample_size_s
## boxplot(gw_distance~sample_size_s,results_all)
## boxplot(gw_distance~sample_size_s,filter(results_all,graphon_id_s==1))

library(tidyr)
library(patchwork)
plot_list = list()  # To store individual plots
# graphon_id_list_s = c(2,4,6,7,8,9,12,13,18,20)
for(i_s in 1:length(graphon_id_list_s)){
  # for(i_t in graphon_id_list_t){
  temp = filter(results_all,graphon_id_s %in% graphon_id_list_s[i_s],
                graphon_id_t %in% graphon_id_list_s[i_s] )
  # Load necessary libraries
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  
  gw_distance = temp$gw_distance
  temp = temp[,-which("gw_distance" %in% colnames(temp))]
  
  # Reshape the data to long format for easier plotting
  data_long = pivot_longer(
    temp,
    ## cols = c(NS, ICE, USVT, SAS, GTrans,GTrans2,Notrans_DS), 
    cols = c("NS", "ICE", "USVT", "SAS", 
             "GTrans_0.01", "GTrans2_0.01", 
             "GTrans_1e.5", "GTrans2_1e.5", 
             "Notrans_DS"),
    
    names_to = "Method",                  # Name of the new column for methods
    values_to = "Error"                   # Name of the new column for error values
  )
  
  # Calculate average error for each sample_size_s and Method
  # averaged_data = data_long %>%
  #   group_by(sample_size_s, Method) %>%
  #   summarize(Average_Error = mean(Error, na.rm = TRUE), .groups = "drop")
  
  averaged_data = data_long %>%
    group_by(sample_size_s, Method) %>%
    summarize(
      Average_Error = mean(Error, na.rm = TRUE),
      SE = sd(Error, na.rm = TRUE),
      .groups = "drop"
    )
  
  
  # Create the line chart with averaged errors
  # averaged_data$Method = factor(averaged_data$Method,
  #                               levels=c("GTrans2","NS","USVT","ICE","SAS","Notrans_DS","GTrans"))
  averaged_data$Method = factor(averaged_data$Method, 
                                levels = c( "GTrans_0.01", 
                                            "GTrans_1e.5", 
                                            "NS", "USVT", "ICE", "SAS", "Notrans_DS","GTrans2_0.01","GTrans2_1e.5"))
  
  
  averaged_data = filter(averaged_data,! Method %in% "Notrans_DS")
  ## averaged_data = filter(averaged_data,! Method %in% "GTrans")
  averaged_data = filter(averaged_data,! Method %in% "GTrans2_1e.5")
  averaged_data = filter(averaged_data,! Method %in% "GTrans2_0.01")

  averaged_data$Method <- recode(averaged_data$Method,
                                 "GTrans_0.01" = "GTrans-EGW",
                                 "GTrans_1e.5" = "GTrans-GW")
  
  
  plot = ggplot(averaged_data, aes(x = sample_size_s, y = Average_Error,
                                   color = Method,
                                   shape=Method,
                                   linetype= Method,
                                   group = Method))  +
    geom_line(size = 1.3) +               # Line connecting averaged points
    geom_point(size = 3, alpha = 1) +  # Add points for averaged errors
    # geom_errorbar(aes(ymin = Average_Error - SE, ymax = Average_Error + SE),
    #               width = 0.01, size = 0.8) +
    theme_bw() +
    labs(
      # title = paste0(i_s,",",i_t),
      x = "Source Sample Size",y = "MSE") +
    # theme(legend.position = "bottom", legend.direction = "horizontal",
    #       axis.title.x = element_text(size = 16),
    #       axis.title.y = element_text(size = 16),
    #       axis.text = element_text(size = 14))+
    # theme(legend.position = "bottom",
    #       legend.title = element_blank(),    # Remove legend title
    #       axis.title.x = element_text(size = 16),
    #       axis.title.y = element_text(size = 16),
    # ) 
    theme(
      legend.position =  "none",    
      legend.title = element_blank(),
      legend.text = element_text(size = 12),
      legend.direction = "horizontal",      
      legend.box = "horizontal" ,
      axis.title.x = element_text(size = 16),  
      axis.title.y = element_text(size = 16), 
      axis.text.x = element_text(size = 14),   
      axis.text.y = element_text(size = 14)    
    ) +
    guides(color = guide_legend(nrow = 1), 
           shape = guide_legend(nrow = 1),
           linetype = guide_legend(nrow = 1)) + 
   
  scale_shape_manual(values = c("GTrans-GW" = 16,    
                                "GTrans-EGW" = 16,   
                                "NS" = 15,           
                                "USVT" = 17,         
                                "ICE" = 18,          
                                "SAS" = 0            
  )) + 
    scale_color_manual(values = c("GTrans-GW" = "#C75736",   
                                  "GTrans-EGW" = deep_pink, 
                                  "NS" = "#3A6DA8",
                                  "USVT" = "#F2C201",
                                  "ICE" = "#7A9832",
                                  "SAS" = "grey")) + 

  scale_linetype_manual(values = c("GTrans-GW" = "solid",
                                   "GTrans-EGW" = "dashed",
                                   "NS" = "solid",
                                   "USVT" = "solid",
                                   "ICE" = "solid",
                                   "SAS" = "solid"))
    
  
  
  plot_list = append(plot_list, list(plot))

}
plot_list[[1]]
library(gridExtra)

png("graphon_N_2cases_V1.png", width = 4000, height = 1500, res = 200)
grid.arrange(grobs = plot_list, ncol = 5, nrow = 2)  
dev.off()



