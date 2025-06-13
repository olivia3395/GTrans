
Double_smooth = function(delta){
  #### delta = phat_t_trasport
  D = aux_nbdsmooth(delta, nrow(delta))
  N = nrow(D)
  h = sqrt(log(N)/N)
  kernel_mat = matrix(0,N,N)
  for (i in 1:N){
    kernel_mat[i,] = as.double(D[i,]<quantile(D[i,],h))
  }
  # L1 normalization of each row
  kernel_mat = kernel_mat/(outer(rowSums(kernel_mat),rep(1,N))+1e-10)
  #Compute P
  P = kernel_mat %*% delta;
  delta_hat  = (P+t(P))/2
  diag(delta_hat) = 0
  # phat_notrans_DS[phat_notrans_DS<0] = 0
  # phat_notrans_DS[phat_notrans_DS>1] = 1
  return(delta_hat)
  
  # A = delta
  # n = nrow(A)
  # # Step 1: Compute A_sq = A %*% A / n
  # A_sq = (A %*% A) / n
  # # Step 2: Structural dissimilarity matrix D
  # D = matrix(0, n, n)
  # for (i in 1:(n - 1)) {
  #   for (j in (i + 1):n) {
  #     diff = abs(A_sq[i, ] - A_sq[j, ])
  #     diff[i] = 0
  #     diff[j] = 0
  #     D[i, j] = D[j, i] = max(diff)
  #   }
  # }
  # # Step 3: Bandwidth h (theoretical default)
  # h = sqrt(log(n) / n)
  # 
  # # Step 4: Neighborhood smoothing
  # P_hat = matrix(0, n, n)
  # for (i in 1:n) {
  #   threshold = quantile(D[i, ], h)
  #   neighbors = which(D[i, ] < threshold)
  #   if (length(neighbors) == 0) {
  #     neighbors = i
  #   }
  #   P_hat[i, ] = colMeans(A[neighbors, , drop = FALSE])
  # }
  # 
  # # Step 5: Symmetrize
  # P_hat = (P_hat + t(P_hat)) / 2
  # delta_hat = P_hat
  # return(delta_hat)
  
}
Graphon_Trans = function(A_s, A_t,threshold=0.2,epsilon,normlization=FALSE){
  density_s = mean(A_s)
  density_t = mean(A_t)
  
  ####NS estimation  for target  data only
  estimate_t = est.nbdsmooth(A_t)
  phat_t = estimate_t$P
  phat_ns = phat_t
  ##double smoothing######
  # phat_t = Double_smooth(phat_t)
  
  ####NS estimation  for source  data
  estimate_s = est.nbdsmooth(A_s)
  phat_s = estimate_s$P
  
  
  # ####LG
  # estimate_t = est.LG(A_t)
  # phat_t = estimate_t$P
  # ####NS estimation  for source  data
  # estimate_s = est.LG(A_s)
  # phat_s = estimate_s$P
  
  # ####ICE
  # source("ICE.R")
  # ice_res = ICE(A = A_t, C_it = 0.2, C_est = 1, P_hat_0 = NULL, rounds = 10)
  # phat_t = ice_res$final_P_hat
  # ice_res = ICE(A = A_s, C_it = 0.2, C_est = 1, P_hat_0 = NULL, rounds = 10)
  # phat_s = ice_res$final_P_hat
  
  diag(phat_t) = 0
  diag(phat_s) = 0
  ############### Combine source and target via OT ################
  # Convert phat_s and phat_t explicitly to NumPy arrays
  # suppose Ps and Pt are your two probability matrices
  M = max(phat_s, phat_t)         # global maximum entry across both matrices
  # now lies in [0,1]
  
  # C1 = np$array( (phat_s - min(phat_s) )/ (max(phat_s) - min(phat_s)))
  # C2 = np$array((phat_t - min(phat_t) )/ (max(phat_t) - min(phat_t)))
  
  # Compute singular values
  svd_A = svd(phat_s)
  # Operator norm is the largest singular value
  if(normlization){
    op_norm_s = max(svd_A$d)
    phat_s_norm = phat_s/(nrow(phat_s))
    phat_t_norm = phat_t/(nrow(phat_s))
    C1 = np$array(phat_s_norm)
    C2 = np$array(phat_t_norm)
  }else{
    phat_s_norm = phat_s
    phat_t_norm = phat_t
    C1 = np$array(phat_s)
    C2 = np$array(phat_t)
  }
  
  
  # Uniform mass distributions
  ns = nrow(phat_s)
  nt = nrow(phat_t)
  p = rep(1/ns, ns)
  q = rep(1/nt, nt)
  
  # Convert mass distributions to NumPy arrays
  p = np$array(p)
  q = np$array(q)
  # Compute GW distance and optimal coupling simultaneously
  
  if(epsilon <=1e-5){
    gw_result = pot$gromov$gromov_wasserstein2(
      # entropic_gromov_wasserstein2
      # gromov_wasserstein2
      C1, C2, p, q, loss.fun = "square_loss",
      epsilon = epsilon, verbose = FALSE, log = TRUE
    )
  }else{
    gw_result = pot$gromov$entropic_gromov_wasserstein2(
      # entropic_gromov_wasserstein2
      # gromov_wasserstein2
      C1, C2, p, q, loss.fun = "square_loss",
      epsilon = epsilon, verbose = FALSE, log = TRUE
    )}
  
  # Extract GW distance and optimal coupling matrix
  gw_distance_squared = gw_result[[1]]
  gw_coupling = gw_result[[2]]$T
  
  # Convert optimal coupling to R matrix
  gw_res_matrix = py_to_r(gw_coupling)
  # Compute GW distance
  gw_distance = sqrt(gw_distance_squared)
  
  # # Compute Gromov-Wasserstein Optimal Transport
  # gw_res = pot$gromov$gromov_wasserstein(
  #   C1, C2, p, q,
  #   loss.fun = 'square_loss', 
  #   epsilon = 1e-5,
  #   verbose = TRUE
  # )
  # Convert result back to R matrix
  # gw_res_matrix = py_to_r(gw_res)
  
  # Display the result
  print(dim(gw_res_matrix))
  print(gw_res_matrix)
  dim(gw_res_matrix)
  
  gw_res_matrix_norm =gw_res_matrix/rowSums(gw_res_matrix)
  gw_res_matrix_norm = gw_res_matrix_norm/colSums(gw_res_matrix_norm)
  
  
  # visualize result
  # library(pheatmap)
  # pheatmap(gw_res_matrix, 
  #          cluster_rows = FALSE, 
  #          cluster_cols = FALSE,
  #          color = viridis::viridis(100),
  #          main = "Gromov-Wasserstein Optimal Transport")
  # 
  phat_t_trasport = t(gw_res_matrix_norm) %*% phat_s %*% gw_res_matrix_norm
  diag(phat_t_trasport) = 0
  
  ##double smoothing######
  phat_t_trasport = Double_smooth(phat_t_trasport)
  # phat_t_trasport = Double_smooth(phat_t_trasport)
  if(epsilon <=1e-5){
    
  density_s = mean(A_s)
  density_t = mean(A_t)
  # Perform the two‐sample proportion test
  result = prop.test(
    x     = c(sum(A_s), sum(A_t)),     # counts of successes
    n     = c(nrow(A_s)*nrow(A_s), nrow(A_t)*nrow(A_t)),     # sample sizes
    alternative = "two.sided",  # "two.sided", "less" or "greater"
    correct     = FALSE        # set to TRUE to apply Yates' continuity correction
  )
  if(result$p.value<0.05 & density_s < density_t){
    threshold = 0.1
  }else{threshold = 0.15}
  }
  else{
    density_s = mean(A_s)
    density_t = mean(A_t)
    # Perform the two‐sample proportion test
    result = prop.test(
      x     = c(sum(A_s), sum(A_t)),     # counts of successes
      n     = c(nrow(A_s)*nrow(A_s), nrow(A_t)*nrow(A_t)),     # sample sizes
      alternative = "two.sided",  # "two.sided", "less" or "greater"
      correct     = FALSE        # set to TRUE to apply Yates' continuity correction
    )
    if(result$p.value<0.05 & density_s < density_t){
      threshold = 0.12
    }else{threshold = 0.18}
  }
  
  
  print(threshold)
  
  
  ######step3: debias step######
  if(gw_distance > threshold){
    phat1 = phat_t_trasport
    # phat1 = P_t
    diag(phat1) = 0
    # delta = A_t  - phat1
    delta = phat_t  - phat1
    
    delta_hat = Double_smooth(delta)
    # delta_hat = Double_smooth(delta_hat)
    # delta_hat = Double_smooth(delta_hat)
    
    # # delta = phat1 - phat_t
    # D = aux_nbdsmooth(delta, nrow(delta))
    # # D = aux_nbdsmooth(A_t, nrow(A_t))
    # 
    # N = nrow(D)
    # h = sqrt(log(N)/N) 
    # kernel_mat = matrix(0,N,N)
    # for (i in 1:N){
    #   kernel_mat[i,] = as.double(D[i,]<quantile(D[i,],h))
    # }
    # # L1 normalization of each row
    # kernel_mat = kernel_mat/(outer(rowSums(kernel_mat),rep(1,N))+1e-10)
    # 
    # #Compute P
    # P = kernel_mat %*% delta;
    # delta_hat  = (P+t(P))/2
    
    # debias: 
    phat_final = delta_hat+ phat1
    # phat_final = phat_t - delta_hat
    # phat_final = delta_hat+ phat_t
    # phat_final = phat1
  }else{
    phat_final = phat_t_trasport
  }
  diag(phat_final) = 0
  phat_final[phat_final<0] = 0 
  phat_final[phat_final>1] = 1 
  
  
  
  ######ablation study: no transfer but with double smoothing######
  phat_notrans_DS = Double_smooth(phat_ns)
  
  
  return(list(phat_s=phat_s,phat_final=phat_final,phat_t_trasport=phat_t_trasport,
              phat_ns=phat_t, phat_notrans_DS=phat_notrans_DS, gw_distance=gw_distance,
              gw_res_matrix_norm= gw_res_matrix_norm,
              phat_s_norm =phat_s_norm, phat_t_norm=phat_t_norm))
}


# Function to calculate the Mean Squared Error (MSE) between two matrices
calculate_mse = function(P1, P2) {
  # Ensure dimensions match
  # if (!all(dim(P1) == dim(P2))) {
  #   stop("Matrices P1 and P2 must have the same dimensions.")
  # }
  
  # Calculate MSE
  # distance = mean((P1 - P2)^2)
  
  ##wasserstein distance ######
  
  ############### Combine source and target via OT ################
  # Convert phat_s and phat_t explicitly to NumPy arrays
  C1 = np$array(P1)
  C2 = np$array(P2)
  
  # Uniform mass distributions
  ns = nrow(P1)
  nt = nrow(P2)
  p = rep(1/ns, ns)
  q = rep(1/nt, nt)
  
  # Convert mass distributions to NumPy arrays
  p = np$array(p)
  q = np$array(q)
  # Compute GW distance and optimal coupling simultaneously
  gw_result = pot$gromov$gromov_wasserstein2(
    C1, C2, p, q, loss.fun = "square_loss",
    epsilon = 1e-5, verbose = TRUE, log = TRUE
  )
  distance = sqrt(gw_result[[1]])
  
  # A = P1
  # eig_out = eigen(A, symmetric = TRUE)
  # values = eig_out$values
  # values[values<0] = 0
  # p=20
  # p = min(sum(values>0),p)
  # idx_top = order(values, decreasing = TRUE)[1:p]
  # eigenvectors_top = eig_out$vectors[, idx_top, drop = FALSE]
  # eigenvalues_top = values[idx_top]  # Top p eigenvalues
  # # Compute sqrt(lambda_i) * eigenvector_i for each top eigenvalue-eigenvector pair
  # embed1 = t(t(eigenvectors_top) * sqrt(eigenvalues_top))
  # 
  # A = P2
  # eig_out = eigen(A, symmetric = TRUE)
  # values = eig_out$values
  # values[values<0] = 0
  # idx_top = order(values, decreasing = TRUE)[1:p]
  # eigenvectors_top = eig_out$vectors[, idx_top, drop = FALSE]
  # eigenvalues_top = values[idx_top]  # Top p eigenvalues
  # # Compute sqrt(lambda_i) * eigenvector_i for each top eigenvalue-eigenvector pair
  # mse =  mean((embed1 - embed2)^2)
  
  return(distance)
}




Graphon_Trans_Ablation = function(A_s, A_t, threshold, epsilon, normlization = FALSE) {

  density_s = mean(A_s)
  density_t = mean(A_t)
  # Perform the two‐sample proportion test
  result = prop.test(
    x     = c(sum(A_s), sum(A_t)),     # counts of successes
    n     = c(nrow(A_s)*nrow(A_s), nrow(A_t)*nrow(A_t)),     # sample sizes
    alternative = "two.sided",  # "two.sided", "less" or "greater"
    correct     = FALSE        # set to TRUE to apply Yates' continuity correction
  )
  if(result$p.value<0.05 & density_s < density_t){
    threshold = 0.1
  }else{threshold = 0.15}
  
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
  
  # C1 = np$array( (phat_s - min(phat_s) )/ (max(phat_s) - min(phat_s)))
  # C2 = np$array((phat_t - min(phat_t) )/ (max(phat_t) - min(phat_t)))
  
  # Compute singular values
  svd_A = svd(phat_s)
  # Operator norm is the largest singular value
  if(normlization){
    op_norm_s = max(svd_A$d)
    phat_s_norm = phat_s/(nrow(phat_s))
    phat_t_norm = phat_t/(nrow(phat_s))
    C1 = np$array(phat_s_norm)
    C2 = np$array(phat_t_norm)
  }else{
    phat_s_norm = phat_s
    phat_t_norm = phat_t
    C1 = np$array(phat_s)
    C2 = np$array(phat_t)
  }
  
  
  # Uniform mass distributions
  ns = nrow(phat_s)
  nt = nrow(phat_t)
  p = rep(1/ns, ns)
  q = rep(1/nt, nt)
  
  # Convert mass distributions to NumPy arrays
  p = np$array(p)
  q = np$array(q)
  # Compute GW distance and optimal coupling simultaneously
  
  
  gw_result = pot$gromov$gromov_wasserstein2(
    # entropic_gromov_wasserstein2
    # gromov_wasserstein2
    C1, C2, p, q, loss.fun = "square_loss",
    epsilon = epsilon, verbose = FALSE, log = TRUE
  )
  
  # Extract GW distance and optimal coupling matrix
  gw_distance_squared = gw_result[[1]]
  gw_coupling = gw_result[[2]]$T
  
  # Convert optimal coupling to R matrix
  gw_res_matrix = py_to_r(gw_coupling)
  # Compute GW distance
  gw_distance = sqrt(gw_distance_squared)
  
  # # Compute Gromov-Wasserstein Optimal Transport
  # gw_res = pot$gromov$gromov_wasserstein(
  #   C1, C2, p, q,
  #   loss.fun = 'square_loss', 
  #   epsilon = 1e-5,
  #   verbose = TRUE
  # )
  # Convert result back to R matrix
  # gw_res_matrix = py_to_r(gw_res)
  
  
  gw_res_matrix_norm =gw_res_matrix/rowSums(gw_res_matrix)
  gw_res_matrix_norm = gw_res_matrix_norm/colSums(gw_res_matrix_norm)
  
    
  phat_t_trasport = t(gw_res_matrix_norm) %*% phat_s %*% gw_res_matrix_norm
  diag(phat_t_trasport) = 0
  
  # 🔹 Variant 1: Double Smooth without Debias (Non-Debias)
  phat_t_trasport_nondebias = Double_smooth(phat_t_trasport)
  
  # 🔹 Variant 2: No Double Smooth but with Debias
  if (gw_distance > threshold) {
    message("Applying debiasing (no smoothing)...")
    phat1 = phat_t_trasport
    diag(phat1) = 0
    
    # Residual and smoothing (debiasing)
    delta = phat_t - phat1
    delta_hat = Double_smooth(delta)  
    
    # Final debiased transport matrix (no Double_smooth on transport itself)
    phat_t_trasport_nosmooth = delta_hat + phat1
  } else {
    message("Threshold not reached (no smoothing), using phat_t_trasport directly.")
    phat_t_trasport_nosmooth = phat_t_trasport
  }
  
  # 🔹 Variant 3: Original Double Smoothing + Debiasing
  if (gw_distance > threshold) {
    message("Applying debiasing on Double Smooth result...")
    phat1 = phat_t_trasport_nondebias
    diag(phat1) = 0
    
    delta = phat_t - phat1
    delta_hat = Double_smooth(delta)
    
    # Final debiased transport matrix
    phat_final = delta_hat + phat1
  } else {
    message("Threshold not reached, using phat_t_trasport_nondebias directly.")
    phat_final = phat_t_trasport_nondebias
  }
  
  # Enforce bounds and clean up diagonal
  diag(phat_final) = 0
  phat_final[phat_final < 0] = 0
  phat_final[phat_final > 1] = 1
  
  ######ablation study: no transfer but with double smoothing######
  phat_notrans_DS = Double_smooth(phat_ns)
  
  ################### New Variant: Raw Adjacency Matrix ###################
  
  # message("🔹 Using raw adjacency matrix instead of smoothed estimate for GW computation.")
  # 
  # # Compute GW with raw adjacency matrices
  # C1_raw = np$array(A_s)
  # C2_raw = np$array(A_t)
  # 
  g_s = graph_from_adjacency_matrix(A_s, mode = "undirected")
  g_t = graph_from_adjacency_matrix(A_t, mode = "undirected")
  
 
  dist_mat_s = distances(g_s)
  dist_mat_t = distances(g_t)
  
  
  dist_mat_s[is.infinite(dist_mat_s)] <- max(dist_mat_s[is.finite(dist_mat_s)]) + 1
  dist_mat_t[is.infinite(dist_mat_t)] <- max(dist_mat_t[is.finite(dist_mat_t)]) + 1
  
  C1_raw = np$array(dist_mat_s)
  C2_raw = np$array(dist_mat_t)
  
  # Uniform mass distributions
  ns = nrow(A_s)
  nt = nrow(A_t)
  p_raw = rep(1/ns, ns)
  q_raw = rep(1/nt, nt)
  
  # Convert to NumPy
  p_raw = np$array(p_raw)
  q_raw = np$array(q_raw)
  
  # Compute GW distance with adjacency matrices
  gw_result_raw = pot$gromov$gromov_wasserstein2(
    C1_raw, C2_raw, p_raw, q_raw, 
    loss.fun = "square_loss", 
    epsilon = epsilon, verbose = FALSE, log = TRUE
  )
  

  # Extract results
  gw_distance_squared_raw = gw_result_raw[[1]]
  gw_coupling_raw = gw_result_raw[[2]]$T
  gw_res_matrix_raw = py_to_r(gw_coupling_raw)
  # Compute the final transport result
  gw_distance_raw = sqrt(gw_distance_squared_raw)
  cat("\n===== GW Distance (Raw Adjacency):", gw_distance_raw, "=====\n")
  
  
  
  # Normalize coupling matrix
  gw_res_matrix_norm_raw = gw_res_matrix_raw/rowSums(gw_res_matrix_raw)
  gw_res_matrix_norm_raw = gw_res_matrix_norm_raw/colSums(gw_res_matrix_norm_raw)
  
  # Compute the final transport result
  phat_t_trasport_raw = t(gw_res_matrix_norm_raw) %*% A_s %*% gw_res_matrix_norm_raw
  diag(phat_t_trasport_raw) = 0
  
  # Apply Double Smoothing without debias (Non-Debias)
  phat_t_trasport_nondebias_raw = Double_smooth(phat_t_trasport_raw)
  
  # Apply debiasing
  if (gw_distance_raw > threshold) {
    message("Applying debiasing on raw adjacency matrix...")
    phat1_raw = phat_t_trasport_raw
    diag(phat1_raw) = 0
    
    # Compute the residual and apply smoothing
    delta_raw = phat_t - phat1_raw
    delta_hat_raw = Double_smooth(delta_raw)
    
    # Final debiased transport matrix
    phat_final_raw_adj = delta_hat_raw + phat1_raw
  } else {
    message("Threshold not reached, using phat_t_trasport_raw directly.")
    phat_final_raw_adj = phat_t_trasport_raw
  }
  
  # Enforce bounds and clean up diagonal
  diag(phat_final_raw_adj) = 0
  phat_final_raw_adj[phat_final_raw_adj < 0] = 0
  phat_final_raw_adj[phat_final_raw_adj > 1] = 1
  
  
  
  
  # Return all variants for comparison
  return(list(
    phat_s = phat_s,
    phat_final = phat_final,
    phat_final_raw_adj = phat_final_raw_adj, 
    phat_t_trasport = phat_t_trasport,
    phat_t_trasport_nondebias = phat_t_trasport_nondebias,
    phat_t_trasport_nosmooth = phat_t_trasport_nosmooth,
    phat_ns = phat_t,
    phat_notrans_DS = phat_notrans_DS,
    gw_distance = gw_distance,
    gw_distance_raw = gw_distance_raw, 
    gw_res_matrix_norm = gw_res_matrix_norm,
    gw_res_matrix_norm_raw = gw_res_matrix_norm_raw,
    phat_s_norm = phat_s_norm,
    phat_t_norm = phat_t_norm
  ))
}
