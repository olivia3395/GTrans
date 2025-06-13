rm(list=ls())
library(graphon)
library(dplyr)
library(ggplot2)
source("network_generate.R")
# source("functions.R")
library(pheatmap)
library(viridis)
library(parallel)
library(gridExtra)
library(grid)
library(graphon)
library(tidyr)

source("auxiliary.R")
source("function.R")
bk = seq(0, 1, 0.01)

library(igraph)
library(transport)
library(gtools)
library(reticulate)
# install.packages("reticulate")
# library(reticulate)
# # Install Python Optimal Transport (POT) library
# py_install("POT", pip = TRUE)


# Ensure Python dependencies
pot = import("ot")
np = import("numpy")

sample_size_t = 50
graphon_id_list_s  =  1:10
graphon_id_list_t = 1:10
sample_size_s_list = seq(100,500,100)
nrepeat = 10
# nrepeat = 10
cores = 20
results_all = NULL
type = "unif"

for(graphon_id_s in graphon_id_list_s){
  
  for(graphon_id_t in graphon_id_list_t){
    # for(graphon_id_t in graphon_id_list_t){
    for(sample_size_s in 500){
      
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
        # result = Graphon_Trans(A_s, A_t, epsilon = 1e-5)
        # phat_final = result$phat_final
        # phat_t_trasport = result$phat_t_trasport
        # phat_t = result$phat_ns
        # phat_notrans_DS= result$phat_notrans_DS
        # gw_distance= result$gw_distance
        # phat_s=result$phat_s
        # ####Transfer learning##########
        # 
        # 
        # error_debias = mean((phat_final  - P_t)^2)
        # error_trans = mean((phat_t_trasport - P_t)^2)
        # error_NS =mean((phat_t - P_t)^2)
        ####error_notrans_DS =mean((phat_notrans_DS - P_t)^2)
        #### Transfer learning (GTrans-GW) ##########
        result_gw = Graphon_Trans(A_s, A_t, epsilon = 1e-5)
        phat_final_gw = result_gw$phat_final
        phat_t_trasport_gw = result_gw$phat_t_trasport
        phat_t_gw = result_gw$phat_ns
        phat_notrans_DS_gw = result_gw$phat_notrans_DS
        gw_distance_gw = result_gw$gw_distance
        phat_s_gw = result_gw$phat_s
        
        
        error_debias_gw = mean((phat_final_gw - P_t)^2)
        error_trans_gw = mean((phat_t_trasport_gw - P_t)^2)
        error_NS = mean((phat_t_gw - P_t)^2)
        
        # cat("GTrans-GW Results:\n")
        # cat("Debias MSE:", error_debias_gw, "\n")
        # cat("Transport MSE:", error_trans_gw, "\n")
        # cat("Neighborhood Smoothing MSE:", error_NS, "\n")
        
        #### Transfer learning (GTrans-EGW) ##########
        result_egw = Graphon_Trans(A_s, A_t, epsilon = 0.01)
        phat_final_egw = result_egw$phat_final
        phat_t_trasport_egw = result_egw$phat_t_trasport
        phat_t_egw = result_egw$phat_ns
        gw_distance_egw = result_egw$gw_distance
        phat_s_egw = result_egw$phat_s
        
        
        error_debias_egw = mean((phat_final_egw - P_t)^2)
        error_trans_egw = mean((phat_t_trasport_egw - P_t)^2)
        
        
        cat("\nGTrans-EGW (ε = 0.01) Results:\n")
        cat("Debias MSE:", error_debias_egw, "\n")
        cat("Transport MSE:", error_trans_egw, "\n")
        
        
              
       
        result = data.frame(
          gw_distance_gw = gw_distance_gw,
          gw_distance_egw = gw_distance_egw,
          NS = error_NS, 
          ICE = error_ICE,
          USVT = error_USVT,
          SAS = error_SAS, 
          GTrans2_GW = error_trans_gw,          
          GTrans_GW = error_debias_gw,          
          GTrans2_EGW = error_trans_egw,     
          GTrans_EGW = error_debias_egw      
        )
        
             
        return(result)
      },mc.cores = cores)
      
      results = do.call("rbind", repeat_results)
      
      
      results <- cbind(results, graphon_id_s, graphon_id_t, sample_size_s)
      results <- as.data.frame(results)
      
     
      results_all = rbind(results_all, results)
      
      
      save(results_all, file = "results_all_cross_graphon_V1.rda")
      
     
      
      data_long = pivot_longer(
        results_all,
        cols = c("NS", "ICE", "USVT", "SAS", 
                 "GTrans_GW", "GTrans2_GW", 
                 "GTrans_EGW", "GTrans2_EGW"), 
        names_to = "Method",
        values_to = "Error"
      )
      
      
      
      
      data_agg = data_long %>%
        group_by(sample_size_s, Method, graphon_id_s, graphon_id_t) %>%
        summarize(
          Average_Error = mean(Error, na.rm = TRUE),
          SE = sd(Error, na.rm = TRUE),
          .groups = "drop"
        )
      
      # Reshape the data using tidyr::pivot_wider to get each method as a column
      reshaped_data = data_agg %>%
        pivot_wider(names_from = Method, values_from = c(Average_Error, SE)) %>% round(3)
      # Print the reshaped data
      print(round(reshaped_data))
      
      write.csv(reshaped_data,"results_all_cross_graphon_V1.csv")
    }
    
  }
}

df = read.csv("results_all_cross_graphon_V1.csv")


# Load libraries
library(dplyr)

# Assume df is your existing data
# For illustration, define dummy 'Similarity' and 'Scenario'
Similarity = rep("Same", nrow(df))
Scenario = paste0(df$graphon_id_s," to ", df$graphon_id_t)

# Define methods to include
methods = c("GTrans_GW","GTrans_EGW", "GTrans2_GW","GTrans2_EGW","NS", "USVT", "ICE", "SAS")
latex_colnames = c("$\\our_GW$", "$\\our_EGW$","$\\our_GW(Nondebias)$","$\\our_EGW(Nondebias)$","NS", "USVT", "ICE", "SAS")

# Create formatted values for each method
formatted_methods = lapply(methods, function(m) {
  avg_col = paste0("Average_Error_", m)
  se_col = paste0("SE_", m)
  sprintf("%.1f $\\pm$ %.1f", df[[avg_col]]*100, df[[se_col]]*100)
})

# Combine everything into a final data frame
latex_df = data.frame(
  Similarity = Similarity,
  Scenario = Scenario,
  setNames(formatted_methods, latex_colnames),
  check.names = FALSE
)
head(latex_df)


write.csv(latex_df, "latex_df_results.csv", row.names = FALSE)


save(latex_df, file = "latex_df_results.RData")


latex_df$Scenario
latex_df2 = filter(latex_df, Scenario %in% c("7 to 6","6 to 7",
                                             "8 to 7","7 to 8","3 to 1","1 to 3"))

# Print rows
apply(latex_df2, 1, function(row) {
  cat(paste(row, collapse = " & "), "\\\\\n")
})

latex_df2 = filter(latex_df, Scenario %in% c("6 to 9","9 to 6",
                                             "5 to 10","10 to 5","9 to 2","2 to 9","9 to 8","8 to 9"))

# Print rows
apply(latex_df2, 1, function(row) {
  cat(paste(row, collapse = " & "), "\\\\\n")
})


