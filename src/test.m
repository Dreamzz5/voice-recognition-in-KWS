
%test

test=rand(5,4,2,3);
W=rand(5,4,2,2);
A=test(:,:,:,:).*W(:,:,:,1);
b=sum(sum(A));