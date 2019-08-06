function y = Softmax_2D(x)
    exponents = x - max(x,[],2);
    ex = exp(exponents);
    y  = ex ./ sum(ex,2);
end