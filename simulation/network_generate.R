asSymmetric = function(M){
  M[lower.tri(M)] = 0
  M = M+t(M)
  diag(M) = diag(M)/2
  return(M)
}
#
gp_generate = function(n, setting,type="unif"){
  
  "
  generate networks with graphons
  
  Inputs:
    n: number of vertexes
    setting: id of graphon used
  
  Outputs:
    P: connecting probability matrix
    A: adjacency matrix
  "
  
  if(type=="seq"){x = y = seq(1/n, 1, 1/n)}
  if(type=="unif"){x = y = runif(n,0,1)}
  
  P = matrix(0, n, n)
  for(i in 1:n){
    for(j in 1:n){
      P[i, j] = switch(setting, 
                       # graphon_id_list = c(2,8,9,7,6,4,12,20,13,18)
                       exp(-((x[i]**0.7)+(y[j]**0.7))),#2
                       exp(-(max(x[i],y[j])**0.75)),#8
                       exp(-0.5*(min(x[i],y[j])+sqrt(x[i])+sqrt(y[j]))),#9
                       1/(1+exp(-((max(x[i],y[j])**2)+(min(x[i],y[j])**4)))), #7
                       abs(x[i]-y[j]), #6
                       (x[i]*y[j])/2, #4
                       ((x[i]^2+y[j]^2)/3)*cos(1/(x[i]^2+y[j]^2))+0.15, #12
                       (x[i] + y[j])/3*cos(1/(x[i] + y[j])) + 0.15, # 20,  NS
                       sin(10*pi*(x[i]+y[j]-5))/5+0.5, #13
                       1/4*min(exp(sin(6/((1-x[i])^2+y[j]^2))), exp(sin(6/(x[i]^2+(1-y[j])^2)))), #18
                       
                       
                       
                       
      )
    }
  }
  A = apply(P, c(1, 2), function(x){rbinom(1, 1, x)})
  A = asSymmetric(A)
  
  return(list(P = P, A = A))
}

