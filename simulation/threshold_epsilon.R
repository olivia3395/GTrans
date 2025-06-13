library(dplyr)
library(parallel)
library(tidyr)
library(ggplot2)


nrepeat = 20
cores = 10
results_all = NULL
type = "unif"
epsilon_list = c(0.001, 0.005, 0.01, 0.02, 0.05, 0.1)  
sample_size_s_list = seq(100, 1000, 100)  
graphon_id_list_s = 1:10


sample_size_t = 50

for(graphon_id_s in graphon_id_list_s){
  
  for(graphon_id_t in graphon_id_s){
    # for(graphon_id_t in graphon_id_list_t){
    for(sample_size_s in sample_size_s_list){
    print(c(graphon_id_s,graphon_id_t,sample_size_s))
    
    for (epsilon in epsilon_list) {
      print(paste("Testing epsilon:", epsilon))
      
      
      repeat_results = mclapply(1:nrepeat, function(iii) {
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
        
        
        gp_s = gp_generate(sample_size_s, graphon_id_s, type = type)
        P_s = gp_s$P
        M = matrix(runif(sample_size_s * sample_size_s, min = -0.01, max = 0.01), 
                   nrow = sample_size_s, ncol = sample_size_s)
        P_s = P_s + M
        P_s = (P_s + t(P_s)) / 2
        diag(P_s) = 0
        P_s[P_s > 1] = 1
        P_s[P_s < 0] = 0
        A_s = gmodel.P(P_s, rep = 1, symmetric.out = TRUE)
        diag(A_s) = 0
        
        # ---- Transfer Learning with Graphon_Trans ----
        result = Graphon_Trans(A_s, A_t, epsilon = epsilon)
        
       
        error_final = mean((result$phat_final - P_t)^2)
        
        # ---- Combine into result ----
        data.frame(
          epsilon = epsilon,
          MSE = error_final
        )
        
      }, mc.cores = cores)
      
      
      results = do.call("rbind", repeat_results)
      results$graphon_id_s = graphon_id_s
      results$sample_size_s = sample_size_s
      
      
      print(results)
      results_all = rbind(results_all, results)
      save(results_all, file = "results_epsilon_ablation_egw.rda")
    }
  }
}
}



load("results_epsilon_ablation_egw.rda")






filtered_data <- results_all %>%
  filter(epsilon %in% c(0.001, 0.005, 0.01, 0.05, 0.1)) %>%
  mutate(epsilon = as.factor(epsilon))


color_mapping <- c(
  "0.001" = "#c2bdde",
  "0.005" = "#82afda",
  "0.01" = "#c92423",
  "0.05" = "#9bbf8a",
  "0.1" = "#f79059"
)


plot_list = list()
graphon_id_list_s = unique(filtered_data$graphon_id_s)


for (i_s in graphon_id_list_s) {
  
  temp_data = filtered_data %>% filter(graphon_id_s == i_s)
  
  plot = ggplot(temp_data, aes(x = epsilon, y = MSE, fill = epsilon)) +
    geom_boxplot(outlier.size = 0.8, outlier.alpha = 0.4, width = 0.6, alpha = 0.6) +
    scale_fill_manual(values = color_mapping) +
    labs(
      title = paste0("Graphon ID: ", i_s),
      x = expression(epsilon),
      y = "Mean Squared Error (MSE)"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 12, face = "bold"),
      axis.title = element_text(size = 11),
      axis.text = element_text(size = 10),
      legend.position = "bottom",
      panel.grid = element_blank()
    )
  
  plot_list = append(plot_list, list(plot))
}


plot_list [[1]]

png("graphon_error_boxplot_egw_clean.png", width = 4000, height = 1500, res = 200)
grid.arrange(grobs = plot_list, ncol = 5, nrow = 2)  
dev.off()



