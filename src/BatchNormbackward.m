function [dx,dbeta,dgamma] = BatchNormbackward(dy,v,cache)
% reference:
%   https://kratzert.github.io/2016/02/12understanding-the-gradient-flow-through-the-batch-normalization-layer.html
% def batchnorm_backward(dout,cache):
%#unfold the variables stored in cache
%   xhat,gamma,xmu,ivar,sqrtvar,var,eps = cache
%#get the dimension of the input/output
%   N,D = dout.shape
%#step9
%   dbeta = np.sum(dout,axis=0)
%   dgammax = dout #not necessary,but more understandable
%#step8
%   dgamma = np.sum(dgammax*xhat,axis=0)
%   dxhat = dgammax * gamma
%#step7
%   divar = np.sum(dxhat*xmu, axis=0)
%   dxmu1 = dxhat * ivar
%#step6
%   dsqrtvar = -1 ./(sqrtvar**2) * divar
%#step5
%   dvar = 0.5 * 1 ./ np.sqrt(var+eps) * dsqrtvar
%#step4
%   dsq = 1 ./N * np.ones((N,D)) * dvar
%#step3
%   dxmu2 = 2 * xmu * dsq
%#step2
%   dx1 = (dxmu1 + dxmu2)
%   dmu = -1 * np.sum(dxmu1 + dxmu2,axis=0)
%#step1
%   dx2 = 1 ./ N * np.ones((N,D)) * dmu
%#step0
%   dx = dx1 + dx2
%   return dx,dgamma,dbeta
    eps = 1e-5;
    N = length(dy);
    dbeta = sum(dy);
    dgamma = sum(dy .* cache.x_hat);
    dxhat = dy .* cache.gamma;
    divar = sum(dxhat .* cache.xmu);
    dxmu_1 = dxhat .* cache.ivar;
    dsqrtvar = -1./(v+eps).* divar;
    dvar = 0.5 .* cache.ivar .* dsqrtvar;
    dsq = dvar .* ones(size(dy))./N;
    dxmu_2 = 2 * cache.xmu .* dsq;
    dx1 = dxmu_1 + dxmu_2;
    dmu = -1 * sum(dx1);
    dx2 = dmu .* ones(size(dy))./N;
    dx = dx1 + dx2;
%briefly write
%   eps = 1e-5;
%   N = length(dy);
%   dgamma = sum(dy * cache.x_hat);
%   dbeta = sum(dy);
%   dx_hat = dy * cache.gamma;
%   dsigma = -0.5 * sum(dx_hat * cache.xmu) * cache.ivar / (v+eps);
%   dmu = -sum(dx_hat * cache.ivar) - 2 * dsigma * sum(cache.xmu)/N;
%   dx = dx_hat * cache.ivar + 2.0 * dsigma * cache.xmu /N + dmu / N;

end