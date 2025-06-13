
rm(list=ls())


library(dplyr)
library(tidyr)
library(ggplot2)
library(pheatmap)
library(parallel)
library(reticulate)
library(graphon)

source("network_generate.R")
source("auxiliary.R")
source("function.R")
source("ICE.R")  

np <- import("numpy")
pot <- import("ot")

type <- "seq"  

graphon_id_list <- 1:10
sample_size_s_list <- c(100, 200, 400, 800)

lambda_list <- seq(-0.5, 0.5, by = 0.05)

sample_size_t <- 50
epsilon_candidates <- seq(0.01, 0.5, 0.01)

nrepeat <- 10
cores <- 10


Graphon_Trans_CV <- function(A_s, A_t, P_t_true,
                             epsilon_list = seq(0.01, 0.3, by = 0.01),
                             np = NULL, pot = NULL) {
  
  ####NS estimation  for target  data only
  estimate_t = est.nbdsmooth(A_t)
  phat_t = estimate_t$P
  phat_ns = phat_t
  ##double smoothing######
  # phat_t = Double_smooth(phat_t)
  
  ####NS estimation  for source  data
  estimate_s = est.nbdsmooth(A_s)
  phat_s = estimate_s$P
  
  
  diag(phat_t) = 0
  diag(phat_s) = 0
  ############### Combine source and target via OT ################
  # Convert phat_s and phat_t explicitly to NumPy arrays
  # suppose Ps and Pt are your two probability matrices
  M = max(phat_s, phat_t)         # global maximum entry across both matrices
  # now lies in [0,1]
  C1 = np$array(phat_s)
  C2 = np$array(phat_t)
  
  # Uniform mass distributions
  ns = nrow(phat_s)
  nt = nrow(phat_t)
  p = rep(1/ns, ns)
  q = rep(1/nt, nt)
  
  # Convert mass distributions to NumPy arrays
  p = np$array(p)
  q = np$array(q)
  # Compute GW distance and optimal coupling simultaneously
  
  
  gw_result <- pot$gromov$gromov_wasserstein2(C1, C2, p, q, loss.fun = "square_loss",
                                              epsilon = 1e-5,
                                              verbose = FALSE, log = TRUE)
  # Extract GW distance and optimal coupling matrix
  gw_distance_squared = gw_result[[1]]
  
  gw_coupling = gw_result[[2]]$T
  # Convert optimal coupling to R matrix
  gw_res_matrix = py_to_r(gw_coupling)
  # Compute GW distance
  gw_distance = sqrt(gw_distance_squared)
  cat("gw_distance:", gw_distance, "\n")
  
  
  gw_res_matrix_norm =gw_res_matrix/rowSums(gw_res_matrix)
  gw_res_matrix_norm = gw_res_matrix_norm/colSums(gw_res_matrix_norm)
  
  
  phat_t_trasport = t(gw_res_matrix_norm) %*% phat_s %*% gw_res_matrix_norm
  diag(phat_t_trasport) = 0
  
  ##double smoothing######
  phat_t_trasport = Double_smooth(phat_t_trasport)
  
 
  rmse_by_eps <- sapply(epsilon_list, function(eps) {
    if (gw_distance > eps) {
      phat1 = phat_t_trasport
      diag(phat1) = 0
      delta = phat_t - phat1
      delta_hat = Double_smooth(delta)
      P_final = delta_hat + phat1
    } else {
      P_final <- phat_t_trasport
    }
    diag(P_final) <- 0
    P_final[P_final < 0] <- 0
    P_final[P_final > 1] <- 1
    
    mean((P_t_true - P_final)^2)
  })
  
  
  rmse_diff <- diff(rmse_by_eps)
  

  
  jump_idx <- which(abs(rmse_diff) > 0)
  
  
  if (length(jump_idx) == 0) {
    cat("No significant jump found, using default epsilon\n")
    jump_idx <- length(epsilon_list) - 1
  } else {
   
    cat("Significant jump found at index:", jump_idx, "\n")
  }
  
  
  jump_eps <- epsilon_list[jump_idx]
  cat("Jump epsilon selected:", jump_eps, "\n")
  
    
 
  mse_debias = rmse_by_eps[jump_idx]
  mse_no_debias = rmse_by_eps[jump_idx + 1]
  
  
  cat("mse_debias:", mse_debias, "\n")
  cat("mse_no_debias:", mse_no_debias, "\n")  
  
  
  if (mse_debias < mse_no_debias) {
    
    best_eps <- min(0.15, jump_eps)
    selected_by <- "Debias Effective: Smaller Epsilon"
    cat("Debias improves MSE, choosing smaller epsilon:", best_eps, "\n")
  } else {
    
    best_eps <- max(0.15, jump_eps)
    selected_by <- "Debias Not Effective: Larger Epsilon"
    cat("Debias increases MSE, choosing larger epsilon:", best_eps, "\n")
  }
  
  
  
  cat("✅ GW distance:", round(gw_distance, 4),
      "| Selected ε:", best_eps,
      "| Selection mode:", selected_by, "\n")
  
  
  
  
  # Final estimator using best epsilon
  if (gw_distance > best_eps) {
    phat1 = phat_t_trasport
    # phat1 = P_t
    diag(phat1) = 0
    # delta = A_t  - phat1
    delta = phat_t  - phat1
    delta_hat = Double_smooth(delta)
    P_final = delta_hat+ phat1
    
    
  } else {
    P_final <- phat_t_trasport
    
  }
  diag(P_final) <- 0
  P_final[P_final < 0] <- 0
  P_final[P_final > 1] <- 1
  
  
  phat_notrans_DS <- Double_smooth(phat_ns)
  
  
  return(list(
    phat_s = phat_ns,
    phat_final = P_final,
    phat_t_trasport = phat_t_trasport,
    phat_ns = phat_t,
    phat_notrans_DS = phat_notrans_DS,
    gw_distance = gw_distance,
    best_epsilon = best_eps,
    jump_epsilon = jump_eps,
    # jump_index = jump_idx,
    best_side = selected_by,
    epsilon_candidates = epsilon_list,
    mse_trace = rmse_by_eps
  ))
}








results_all <- list()

for (graphon_id in graphon_id_list) {
  for (sample_size_s in sample_size_s_list) {
    for (lambda in lambda_list) {
      
      repeat_eps <- mclapply(1:nrepeat, function(seed) {
        set.seed(seed)
        
        # Target graph
        P_t <- gp_generate(sample_size_t, graphon_id, type=type)$P
        P_t <- (P_t + t(P_t)) / 2
        diag(P_t) <- 0
        A_t <- gmodel.P(P_t, rep=1,symmetric.out = TRUE)
        diag(A_t) <- 0
        
        # Source graph with noise
        P_s <- gp_generate(sample_size_s, graphon_id, type=type)$P
        noise <- matrix(runif(sample_size_s^2, min = 0, max = abs(lambda)), sample_size_s, sample_size_s)
        noise <- if (lambda >= 0) noise else -noise
        P_s <- P_s + noise
        P_s <- (P_s + t(P_s)) / 2
        diag(P_s) <- 0
        P_s[P_s > 1] <- 1
        P_s[P_s < 0] <- 0
        
        A_s <- gmodel.P(P_s, rep = 1, symmetric.out = TRUE)
        
        diag(A_s) <- 0
        
        
        # Graphon transfer with CV
        res <- Graphon_Trans_CV(A_s, A_t, P_t_true = P_t,
                                epsilon_list = epsilon_candidates,
                                np = np,
                                pot = pot)
        
        
        
        return(res$best_epsilon)
        
        
      }, mc.cores = cores)
      
      
      
      
      ## avg_eps <- mean(unlist(repeat_eps))
      avg_eps <- median(unlist(repeat_eps))
      
      
      
      cat("📌 graphon_id =", graphon_id,
          "| sample_size_s =", sample_size_s,
          "| lambda =", lambda,
          "\n→ best_eps =", paste(round(unlist(repeat_eps), 4), collapse = ", "),
          "\n→ avg_eps =", round(avg_eps, 4), "\n\n")
      
      
      results_all <- append(results_all, list(
        data.frame(
          graphon_id = graphon_id,
          lambda = lambda,
          sample_size_s = sample_size_s,
          best_epsilon = avg_eps
        )
      ))
    }
  }
}

results_df <- bind_rows(results_all)



library(ggplot2)
library(dplyr)
library(tidyr)


print(results_df)

results_df$best_epsilon <- as.numeric(results_df$best_epsilon)
results_df$label_text <- sprintf("%.2f", results_df$best_epsilon)




write.csv(results_df, "results_df_cv.csv", row.names = FALSE)



results_df <- read.csv("results_df_cv.csv")


library(ggplot2)
library(dplyr)
library(tidyr)


results_df$best_epsilon <- as.numeric(results_df$best_epsilon)
results_df$label_text <- sprintf("%.2f", results_df$best_epsilon)


p <- ggplot(results_df, aes(x = factor(sample_size_s), y = lambda, fill = best_epsilon)) +
  geom_tile(color = "white") +
  geom_text(aes(label = label_text), size = 4, color = "black") +
  facet_wrap(~ graphon_id, nrow = 2, ncol = 5) +  
  scale_fill_viridis_c(
    name = "Best δ ",
    option = "C",
    limits = c(0.01, 0.3),
    breaks = c(0.01, 0.1, 0.2, 0.3),
    guide = guide_colorbar(ticks = TRUE)
  ) +
  scale_y_continuous(
    breaks = c(-0.25, 0, 0.25),
    labels = c("-0.25", "0", "0.25")
  ) +
  labs(
    x = "Sample Size (Source)",
    y = expression(lambda)
   #  title = "Best δ  (Threshold) for Different Graphon IDs: GTrans-GW"
  ) +
  theme_minimal(base_size = 14) +
  # theme(
  #   panel.grid = element_blank(),
  #   axis.text.x = element_text(angle = 45, hjust = 1),
  #   strip.text = element_text(size = 12, face = "bold"),
  #   plot.title = element_text(hjust = 0.5, size = 16, face = "bold")
  # )
  theme(
    panel.grid = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 16),    
    axis.text.y = element_text(size = 16),                           
    axis.title.x = element_text(size = 18, face = "bold"),           
    axis.title.y = element_text(size = 18, face = "bold"),           
    strip.text = element_text(size = 12, face = "bold"),
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold")
  )




ggsave("fig_neurips_graphoncv.png", p, width = 7, height = 4, dpi = 300, units = "in")



print(p)


