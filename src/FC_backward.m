function [dx,dw,db] = FC_backward(x,w,delta)
    [batchsize,~] = size(x);
    dx = delta * w';   
    dw = x' * delta ./batchsize;
    db = sum(delta,1)./batchsize;
end