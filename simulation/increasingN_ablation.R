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

graphon_id_list_s  = 1:10
# graphon_id_list_t = graphon_id_list_s


# graphon_id_list_t = graphon_id_list_s
sample_size_s_list = seq(100, 1000, 100)





nrepeat = 50
# nrepeat = 10
cores = 20
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
        
        
        
        
        
        ####Transfer learning##########
        ### result = Graphon_Trans(A_s, A_t,epsilon = 1e-5)
        # ---- epsilon = 0.01 ----
        # ---- epsilon = 1e-5 ----
        result_1 = Graphon_Trans_Ablation(A_s, A_t, epsilon = 1e-5)
        
        
        error_nondebias = mean((result_1$phat_t_trasport_nondebias - P_t)^2)
        error_nosmooth  = mean((result_1$phat_t_trasport_nosmooth - P_t)^2)
        error_final     = mean((result_1$phat_final - P_t)^2)
        
        
        gw_distance = result_1$gw_distance
        
        
        
        error_raw      = mean((result_1$phat_final_raw_adj - P_t)^2)
        gw_distance_raw = result_1$gw_distance_raw
        
        
        cat("[🔍] GW Distance for EGW: ", gw_distance, "\n")
        cat("[📊] Error (Non-Debias, Double Smooth): ", error_nondebias, "\n")
        cat("[📊] Error (No Smooth, Debias): ", error_nosmooth, "\n")
        cat("[📊] Error (Double Smooth + Debias): ", error_final, "\n")
        cat("[🔍] GW Distance (Raw Adjacency): ", gw_distance_raw, "\n")
        cat("[📊] Error (Raw Adjacency Transport): ", error_raw, "\n")
        
        # ---- Combine into result ----
        result = data.frame(
          gw_distance = gw_distance,
          gw_distance_raw = gw_distance_raw,
          GTrans_Nondebias = error_nondebias,
          GTrans_Nosmooth = error_nosmooth,
          GTrans_Final = error_final,
          GTrans_Raw = error_raw
        )
        
        
               
        
        return(result)
        
      },mc.cores = cores)
      
      results = do.call("rbind",repeat_results)
      results$graphon_id_s = graphon_id_s
      results$graphon_id_t = graphon_id_t
      results$sample_size_s = sample_size_s
      print(results)
      results_all = rbind(results_all,results)
      save(results_all,file="results_N_ablation.rda")
    }
    
    library(tidyr)
    library(patchwork)
    plot_list = list()  # To store individual plots
    
    for (i_s in 1:length(graphon_id_list_s)) {
      
      
      results_all_df = as.data.frame(results_all)
      temp = filter(results_all_df, 
                    graphon_id_s %in% graphon_id_list_s[i_s],
                    graphon_id_t %in% graphon_id_list_s[i_s])
      
      
      temp = temp %>%
        select(sample_size_s, GTrans_Nondebias, GTrans_Nosmooth, GTrans_Final, GTrans_Raw)
      
     
      library(ggplot2)
      library(dplyr)
      library(tidyr)
      
      data_long = pivot_longer(
        temp,
        cols = c("GTrans_Nondebias", "GTrans_Nosmooth", "GTrans_Final", "GTrans_Raw"),
        names_to = "Method",
        values_to = "Error"
      )
      
      
      averaged_data = data_long %>%
        group_by(sample_size_s, Method) %>%
        summarize(
          Average_Error = mean(Error, na.rm = TRUE),
          SE = sd(Error, na.rm = TRUE),
          .groups = "drop"
        )
      
      
      averaged_data$Method = factor(averaged_data$Method, 
                                    levels = c("GTrans_Nondebias", "GTrans_Nosmooth", "GTrans_Final", "GTrans_Raw"))
      
      
      plot = ggplot(averaged_data, aes(x = sample_size_s, y = Average_Error, 
                                       color = Method, 
                                       shape = Method,
                                       linetype = Method,
                                       group = Method))  +
        geom_line(size = 1.2) +               
        geom_point(size = 3, alpha = 0.8) +   
        theme_bw() +
        labs(
          x = "Sample Size (s)", y = "Average Error"
        ) +  
        theme(legend.position = "right",
              legend.title = element_blank(),
              panel.grid.minor = element_blank()
        ) +
        
        scale_shape_manual(values = c(8, 9, 16, 17)) + 
        scale_color_manual(values = c("orange", "#33A02C", "#C75736","purple")) +
        scale_linetype_manual(values = c("dashed", "dotted", "solid", "dotdash"))
      
      
      
      plot_list = append(plot_list, list(plot))
    }
    
    library(gridExtra)
    # Save the grid layout manually using png()
    png("graphon_error_N.png", width = 4000, height = 1500, res = 200)
    grid.arrange(grobs = plot_list, ncol = ceiling(length(graphon_id_list_s)/2), nrow = 2)
    dev.off()
    
    
  }
}

# 
load("results_N_ablation.rda")



hist(results_all$gw_distance)
head(results_all)


library(tidyr)
library(patchwork)
library(colorspace)
library(ggplot2)
library(dplyr)
library(gridExtra)



plot_list = list()  # To store individual plots






for(i_s in 1:length(graphon_id_list_s)) {
  
  
  temp = filter(results_all,
                graphon_id_s %in% graphon_id_list_s[i_s],
                graphon_id_t %in% graphon_id_list_s[i_s])
  
  
  temp = temp %>%
    select(sample_size_s, GTrans_Nondebias, GTrans_Nosmooth, GTrans_Final, GTrans_Raw)
  
  
  data_long = pivot_longer(
    temp,
    cols = c("GTrans_Nondebias", "GTrans_Nosmooth", "GTrans_Final","GTrans_Raw"),
    names_to = "Method",
    values_to = "Error"
  )
  
  
  averaged_data = data_long %>%
    group_by(sample_size_s, Method) %>%
    summarize(
      Average_Error = mean(Error, na.rm = TRUE),
      SE = sd(Error, na.rm = TRUE),
      .groups = "drop"
    )
  
 
  averaged_data$Method = factor(averaged_data$Method, 
                                levels = c("GTrans_Nondebias", "GTrans_Nosmooth", "GTrans_Final", "GTrans_Raw"))
  
  
  averaged_data$Method <- recode(averaged_data$Method,
                                 "GTrans_Nondebias" = "GTrans-NonDebias",
                                 "GTrans_Nosmooth" = "GTrans-NonSmooth",
                                 "GTrans_Final" = "GTrans",
                                 "GTrans_Raw" = "GTrans-Adj")
  
 
  plot = ggplot(averaged_data, aes(x =sample_size_s, y = Average_Error,
                                   color = Method,
                                   shape = Method,
                                   linetype = Method,
                                   group = Method)) +
    geom_line(size = 1.3) +               
    geom_point(size = 3, alpha = 1) +  
    theme_bw() +
    labs(
      x = "Sample Size", y = "MSE"
    ) +  
    theme(
      legend.position =  "bottom",
      legend.title = element_blank(),
      legend.text = element_text(size = 12),
      axis.title.x = element_text(size = 16),
      axis.title.y = element_text(size = 16),
      axis.text.x = element_text(size = 14),
      axis.text.y = element_text(size = 14)
    ) +
   
  scale_shape_manual(values = c("GTrans-NonDebias" = 16,   
                                "GTrans-NonSmooth" = 17,     
                                "GTrans" = 18 ,
                                "GTrans-Adj" = 15 
  )) + 
    # scale_color_manual(values = c("GTrans-NonDebias" = "#FDD379",   
    #                               "GTrans-NoSmooth" = "#FA9E39", #"#715ea9", 
    #                               "GTrans-Final" = "#E7483D")) + 
    scale_color_manual(values = c("GTrans-NonDebias" = "#9d84bf",   
                                  "GTrans-NonSmooth" = "#f79059", #"#715ea9", 
                                  "GTrans" = "#c82423",
                                  "GTrans-Adj" ="#FDD379")) + 
    scale_linetype_manual(values = c("GTrans-NonDebias" = "dotted",
                                     "GTrans-NonSmooth" = "dashed",
                                     "GTrans" = "solid",
                                     "GTrans-Adj" ="dotdash"))
  
  
  plot_list = append(plot_list, list(plot))
}

plot_list[[1]]


png("graphon_N_ablation.png", width = 4000, height = 1500, res = 200)
grid.arrange(grobs = plot_list, ncol = 5, nrow = 2)
dev.off()









