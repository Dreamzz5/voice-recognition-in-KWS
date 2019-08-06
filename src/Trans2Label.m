function [d] = Trans2Label(D,classNames,N)
    d = zeros(N,numel(classNames));
    for i = 1:N
        d(i,1:numel(classNames)) = (D(i,1)==classNames);
    end
end