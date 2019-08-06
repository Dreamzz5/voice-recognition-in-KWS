clc;clear;
load Xtrain.mat; %10 49 1 36923

Xtrain=Xtrain(:,:,:,1:3000);
[~,~,~,datasize]=size(Xtrain);
load('train_label.mat') 
load validation_label.mat
commands(:,1) = ["silence","unknown","yes","no","up","down","left","right","on","off","stop","go"];
classNames = categorical(commands);
Label.train   = Trans2Label(train_label,classNames,datasize);
Label.validation   = Trans2Label(validation_label,classNames,datasize);
batchsize=50;
maxvalacc=0;
%第一层卷积
W_conv = 0.1 * randn(4,10,1,32); 	
Bias_conv = zeros(1,32);
stride_h=1;
stride_w=1;
%第二层卷积
W_conv2 = 0.1 * randn(4,10,32,32); 	
Bias_conv2 = zeros(1,32);
stride_h2=1;
stride_w2=1;

%batch_normolization 第一层
batch_normolization.beta1=zeros(1,batchsize);
batch_normolization.gamma1=ones(1,batchsize);

%batch normolization 第二层
batch_normolization.beta2=zeros(1,batchsize);
batch_normolization.gamma2=ones(1,batchsize);
%batch normolization 第三层
batch_normolization.beta3=zeros(1,batchsize);
batch_normolization.gamma3=ones(1,batchsize);
%Fc1
W_FC1 = 0.1 * randn(3968,40); 	
Bias_FC1 = zeros(1,40);
%Fc2
W_FC2 = 0.1 * randn(40,12); 	
Bias_FC2 = zeros(1,12);
%adam

for epoch=1:50
    tacc = zeros(ceil(datasize/batchsize),1);
    tloss= zeros(ceil(datasize/batchsize),1);
    acc=0;t = 1;
    epochtime0 = tic;
for i=1:ceil(datasize/batchsize)
    
    Batchsize=min(batchsize,datasize-(i-1)*batchsize);
    
    Input = Xtrain(:,:,:,(i-1)*batchsize+1:(i-1)*batchsize+Batchsize);  
    %10    49     1   1000
    
    [y_conv1,conv_cache1] = ConvFast(Input,W_conv,Bias_conv,stride_h,stride_w,"valid");
    % 7    40    32   1000
    [x0,y0,z0,~]=size(y_conv1);
    y_conv1=reshape(y_conv1,x0*y0*z0,Batchsize);
    %BatchNorm1
    [BatchNorm1,u1,v1,BatchNorm1_cache1] = BatchNormforward(y_conv1,batch_normolization.beta1,batch_normolization.gamma1);
    BatchNorm1=reshape(BatchNorm1,x0,y0,z0,Batchsize);
    
    % 7    40    32   1000
    ReLUforward1 = ReLU(BatchNorm1);
    [y_conv2,conv_cache2] = ConvFast(ReLUforward1,W_conv2,Bias_conv2,stride_h2,stride_w2,"valid");
    %4    31    32   1000
    [x,y,z,~]=size(y_conv2);
    y_conv2=reshape(y_conv2,x*y*z,Batchsize);
    %BatchNorm2
    [BatchNorm2,u2,v2,BatchNorm2_cache2] = BatchNormforward(y_conv2,batch_normolization.beta2,batch_normolization.gamma2);
    
    BatchNorm2=reshape(BatchNorm2,x,y,z,Batchsize);
 
    %4    31    32   1000
    ReLUforward2 = ReLU(BatchNorm2);
    
    %FC1 
    %4    31    32   1000
    ReLUforward2=reshape(ReLUforward2,x*y*z,Batchsize);
    FC1_forward1=ReLUforward2'*W_FC1+Bias_FC1;
    
    %BatchNorm3
    [BatchNorm3,u3,v3,BatchNorm3_cache3] = BatchNormforward(FC1_forward1',batch_normolization.beta3,batch_normolization.gamma3);
    
    %ReLU
    ReLUforward3 = ReLU(BatchNorm3);
    
    %FC2 
    FC1_forward1=ReLUforward3'*W_FC2+Bias_FC2;
    
    %Softmax
    Softmax_forward1 = Softmax_2D(FC1_forward1);
    
    %%反向

    label = Label.train((i-1)*batchsize+1:(i-1)*batchsize+Batchsize,:);  
    
    err=label-Softmax_forward1;
    delta=-err;
    
    %FC2_backward
    [dx.FC2_backward,dw.FC2_backward,db.FC2_backward] = FC_backward(ReLUforward3',W_FC2,delta);
    
    %ReLU3 backward
    dx.ReLU3=(ReLUforward3>0).*dx.FC2_backward';
    
    %%BatchNorm3
    [dx.BatchNorm3,dbeta.BatchNorm3,dgamma.BatchNorm3] = BatchNormbackward(dx.ReLU3,v3,BatchNorm3_cache3);
    
    %FC1_backward
    [dx.FC1_backward,dw.FC1_backward,db.FC1_backward] = FC_backward(ReLUforward2',W_FC1,dx.BatchNorm3');
    
    %ReLU2 backward
    dx.ReLU2=reshape(dx.FC1_backward',x,y,z,Batchsize);
    dx.ReLU2=(BatchNorm2>0).*dx.ReLU2;
    
    %BatchNorm2
    dx.ReLU2=reshape(dx.ReLU2,x*y*z,Batchsize);
    [dx.BatchNorm2,dbeta.BatchNorm2,dgamma.BatchNorm2] = BatchNormbackward(dx.ReLU2,v2,BatchNorm2_cache2);
    dx.BatchNorm2=reshape(dx.BatchNorm2,x,y,z,Batchsize);
    
    %dconv2
    [dx.conv2,dw.conv2,db.conv2] = Conv2D_backward_2(dx.BatchNorm2,conv_cache2);
    
    %BatchNorm1
    dx.conv2=reshape(dx.conv2,x0*y0*z0,Batchsize);
    [dx.BatchNorm1,dbeta.BatchNorm1,dgamma.BatchNorm1] = BatchNormbackward(dx.conv2,v1,BatchNorm1_cache1);
    dx.BatchNorm1=reshape(dx.BatchNorm1,x0,y0,z0,Batchsize);
    %dconv2
    [dx.conv1,dw.conv1,db.conv1] = Conv2D_backward(dx.BatchNorm1,conv_cache1); 
    %%
    %update paremeters    
        this(1).forward = W_conv;                           
        this(2).forward = W_conv2;
        this(3).forward = W_FC1;
        this(4).forward = W_FC2;
        this(5).forward = batch_normolization.beta1;
        this(6).forward = batch_normolization.beta2;
        this(7).forward = batch_normolization.beta3;
        this(8).forward = batch_normolization.gamma1;
        this(9).forward = batch_normolization.gamma2;
        this(10).forward = batch_normolization.gamma3;
		this(11).forward = Bias_conv;
		this(12).forward = Bias_conv2;
        this(13).forward = Bias_FC1;
        this(14).forward = Bias_FC2;
		
		this(1).back = dw.conv1;
        this(2).back = dw.conv2;
        this(3).back = dw.FC1_backward;
        this(4).back = dw.FC2_backward;
		this(5).back = dbeta.BatchNorm1;
		this(6).back = dbeta.BatchNorm2;
		this(7).back = dbeta.BatchNorm3;
		this(8).back = dgamma.BatchNorm1;
        this(9).back = dgamma.BatchNorm2;
        this(10).back =dgamma.BatchNorm3;
        this(11).back = db.conv1;
		this(12).back = db.conv2;
        this(13).back = db.FC1_backward;
        this(14).back = db.FC2_backward;
   
 %%    run adam
         globalLearnRate = 0.001; %% 学习率
         LocalL2Factors = [ones(1,4),zeros(1,10)];%% 正则化操作 4
         GradientMovingAverage = num2cell(zeros(1,14));
         SquaredGradientMovingAverage = num2cell(zeros(1,14));

 
 
        [backout,GradientMovingAverage,SquaredGradientMovingAverage] ...
        = adam(globalLearnRate,...
          this,....
          LocalL2Factors,...
          GradientMovingAverage,...
          SquaredGradientMovingAverage,...
          t);
      
          W_conv                       = W_conv                    +	backout{1} ;
          W_conv2                      = W_conv2                   +	backout{2} ;
          W_FC1                        = W_FC1                     +	backout{3} ;
          W_FC2                        = W_FC2                     +	backout{4} ;
          batch_normolization.beta1    = batch_normolization.beta1 +	backout{5} ;
          batch_normolization.beta2    = batch_normolization.beta2 +	backout{6} ;
          batch_normolization.beta3    = batch_normolization.beta3 +	backout{7} ;
          batch_normolization.gamma1   = batch_normolization.gamma1+	backout{8} ;
          batch_normolization.gamma2   = batch_normolization.gamma2+	backout{9} ;
          batch_normolization.gamma3   = batch_normolization.gamma3+	backout{10};
          Bias_conv                    = Bias_conv                 +	backout{11};
          Bias_conv2                   = Bias_conv2                +	backout{12};
          Bias_FC1                     = Bias_FC1                  +	backout{13};
          Bias_FC2                     = Bias_FC2                  +	backout{14};
          
          t = t + 1;   
          
    %% compute acc
        dx = [];
        dw = [];
        db = [];
        [~,index_d] = max(label,[],2);
        [~,index_y] = max(Softmax_forward1,[],2);
        tacc(i) = sum(index_d == index_y); 
        tloss(i) = -sum(log(nnet.internal.cnn.util.boundAwayFromZero(Softmax_forward1(find(label == max(label,[],2))))));
        acc_buf = mean(index_d == index_y) %* Batchsize/datasize; 
        %acc = acc + acc_buf
        
end
    epochtime = toc(epochtime0)
    trainacc(epoch) = sum(tacc)/datasize;
    trainLoss(epoch) = sum(tloss)/datasize;
    
       
        load 'Xvalidation.mat'
        Xvalidation=Xvalidation(:,:,:,1:50);
        [~,~,~,N]=size(Xvalidation);
        [valacc(epoch),valLoss(epoch)] = validation(...
                      Xvalidation,Label,N,batchsize,...
                      W_conv,Bias_conv,stride_h,stride_w,...
                      W_conv2,Bias_conv2,stride_h2,stride_w2,...
                      batch_normolization ,W_FC1,W_FC2,Bias_FC1,Bias_FC2);
    if valacc(epoch) >= maxvalacc
        maxvalacc = valacc(epoch);
        maxepoch = epoch;
       save('Trained_WB.mat','W_conv','W_conv2','W_FC1','W_FC2',...
        'batch_normolization','Bias_conv','Bias_conv2','Bias_FC1','Bias_FC2');               
    end
      %% Display Result
    disp('--------------------------------------------------------')
    disp("epoch " + epoch + "    Training loss : " + trainLoss(epoch));
    disp("            Training acc : " + trainacc(epoch)*100 + "%");
    disp("            Validation loss : "+ valLoss(epoch));
    disp("            Validation acc : " + valacc(epoch) *100 + "%");
    f1 = figure(1);
    subplot 211   
    plot(valacc);
    xlabel('allEpoch');
    ylabel('Validation accuracy');
    title('Validation');
    subplot 212
    plot(valLoss);
    xlabel('allEpoch');
    ylabel('Validation Loss');  
    f2 = figure(2);
    subplot 211   
    plot(trainacc);
    xlabel('allEpoch');
    ylabel('Train accuracy');
    title('Train');
    subplot 212
    plot(trainLoss);
    xlabel('allEpoch');
    ylabel('Train Loss');
    drawnow;
end