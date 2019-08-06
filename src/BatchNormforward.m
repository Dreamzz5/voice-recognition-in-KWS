function [o,u,v,cache] = BatchNormforward(x,beta,gamma)
% def batchnorm_forward(x,gamma,beta,eps):
%   N,D = x.shape
%#step1: calculate mean
%   mu = 1./N * np.sum(x,axis=0)
%#step2: subtract mean vector of every trainings example
%   xmu = x -mu
%#step3: following the lower branch - calculation denominator
%   sq = xmu ** 2
%#step4: calculate variance
%   var = 1./N * np.sum(sq,axis = 0)
%step5: add eps for numerical stability,the sqrt
%   sqrtvar = np.sqrt(var + eps)
%#step6: invert sqrtvar
%   ivar = 1./sqrtvar
%#step7: execute normalization
%   xhat = xmu * ivar
%#step8: Nor the two transformation steps
%   gammax = gamma * xhat
%#step9:
%   out = gammax + beta
%#store intermediate
%   cache = (xhat,gamma,xmu,ivar,sqrtvar,var,eps)
%   return out,cache
    eps = 1e-5;
    N = length(x);
    [~,batchsize]=size(x);
    gamma_tmp=gamma(1,1:batchsize);
    beta_tmp=beta(1,1:batchsize);
    u = sum(x) / N;
    xmu = x - u;
    v = sum((xmu.^2)) /N;
    ivar = 1./sqrt(v + eps);
    x_hat = xmu .* ivar;
    o =  gamma_tmp .*x_hat + beta_tmp;
    cache = struct('x_hat',x_hat,'gamma',gamma,'xmu',xmu,'ivar',ivar);
end