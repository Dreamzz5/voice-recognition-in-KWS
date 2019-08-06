function [acc,Loss] = validation(...
                      Xvalidation,Label,N,bsize,...
                      W_conv,Bias_conv,stride_h,stride_w,...
                      W_conv2,Bias_conv2,stride_h2,stride_w2,...
                      batch_normolization ,W_FC1,W_FC2,Bias_FC1,Bias_FC2)                             
    tacc = zeros(ceil(N/bsize),1);
    tloss= zeros(ceil(N/bsize),1);
    for i=1:ceil(N/bsize)
    
    Batchsize=min(bsize,N-(i-1)*bsize);
    
    Input = Xvalidation(:,:,:,(i-1)*bsize+1:(i-1)*bsize+Batchsize);  
    %10    49     1   100
    
    [y_conv1,~] = Conv2D_forward(Input,W_conv,Bias_conv,stride_h,stride_w,"valid");
    % 7    40    32   100
    [x0,y0,z0,~]=size(y_conv1);
    y_conv1=reshape(y_conv1,x0*y0*z0,Batchsize);
    %BatchNorm1
    [BatchNorm1,~,~,~] = BatchNormforward(y_conv1,batch_normolization.beta1,batch_normolization.gamma1);
    BatchNorm1=reshape(BatchNorm1,x0,y0,z0,Batchsize);
    
    % 7    40    32   100
    ReLUforward1 = ReLU(BatchNorm1);
    
    [y_conv2,~] = Conv2D_forward(ReLUforward1,W_conv2,Bias_conv2,stride_h2,stride_w2,"valid");
    %4    31    32   100
    [x,y,z,~]=size(y_conv2);
    y_conv2=reshape(y_conv2,x*y*z,Batchsize);
    %BatchNorm2
    [BatchNorm2,~,~,~] = BatchNormforward(y_conv2,batch_normolization.beta2,batch_normolization.gamma2);
    
    BatchNorm2=reshape(BatchNorm2,x,y,z,Batchsize);
 
    %4    31    32   100
    ReLUforward2 = ReLU(BatchNorm2);
    
    %FC1 
    %4    31    32   100
    ReLUforward2=reshape(ReLUforward2,x*y*z,Batchsize);
    FC1_forward1=ReLUforward2'*W_FC1+Bias_FC1;
    
    %BatchNorm3
    [BatchNorm3,~,~,~] = BatchNormforward(FC1_forward1',batch_normolization.beta3,batch_normolization.gamma3);
    
    %ReLU
    ReLUforward3 = ReLU(BatchNorm3);
    
    %FC2 
    FC1_forward1=ReLUforward3'*W_FC2+Bias_FC2;
    
    %Softmax
    Softmax_forward1 = Softmax_2D(FC1_forward1);
        d = Label.validation((i-1)*bsize + 1:(i-1)*bsize + Batchsize,:);
        [~,index_d] = max(d,[],2);
        [~,index_y] = max(Softmax_forward1,[],2);
        tacc(i) = sum(index_d == index_y); 
        tloss(i) = -sum(log(nnet.internal.cnn.util.boundAwayFromZero(Softmax_forward1(find(d == max(d,[],2))))));
    end   
    acc = sum(tacc)/N;
    Loss = sum(tloss)/N;
end