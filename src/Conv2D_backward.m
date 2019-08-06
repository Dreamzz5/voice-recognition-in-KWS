% function [dw,db,dx1]=BP_Cov_Update(Originalinput,W,dx,...
%     patchl,stridel,...
%     patchh,strideh)
function [dx,dw,db] = Conv2D_backward(dY,cache)
% cache = {Input,W,stride_h,stride_w,MOD_h,MOD_w,W_height,W_width,...
%          ip_channel,op_channel,ip_height,ip_width,batch,flag};
%     W=permute(W,[1,2,4,3]);
%     [W_height,W_width,W_deep,Input_channel] = size(W);
%     [~,~,input_deep,~] = size(Originalinput);
    [dY_height,dY_width,dY_deep,batch] = size(dY);
    W=permute(cache.W,[1,2,4,3]);
    padding_h = cache.W_height-1;
    padding_w = cache.W_width-1;
% Formula:
%   (X_h + 2*padding_h - W_h)./stride_h + 1 = y_h
%   (X_w + 2*padding_w - W_w)./stride_w + 1 = y_w
    S_1_dY_h = (dY_height- 1)*cache.stride_h + 1 + 2*padding_h + cache.MOD_h;
    S_1_dY_w = (dY_width - 1)*cache.stride_w + 1 + 2*padding_w + cache.MOD_w;
    
    delta=zeros(S_1_dY_h,S_1_dY_w,dY_deep,batch);  
% expand dx to the result of which use Stride=1 to cov input
    for i = 1:dY_deep
        for j = 1:dY_width
            for k = 1:dY_height
                delta((k-1)*cache.stride_h+1+padding_h,...
                      (j-1)*cache.stride_w+1+padding_w,...
                       i,:)=dY(k,j,i,:);
            end
        end
    end              
    
    output_h = floor((S_1_dY_h - cache.W_height) + 1);
    output_w = floor((S_1_dY_w - cache.W_width) + 1); 
    dx = zeros(output_h,output_w,cache.ip_channel,batch);
    tic
    W_new=rot90(rot90(W));
    for ip_ch = 1 : cache.ip_channel
        for oh = 1 : output_h
            for ow = 1 : output_w
             tmp = delta(((oh-1)+1):((oh-1)+cache.W_height),...
                         ((ow-1)+1):((ow-1)+cache.W_width),:,:);
             dx(oh,ow,ip_ch,:) = sum(sum(sum(tmp .* W_new(:,:,:,ip_ch))));
            end
        end           
    end             %calculate dx to previous layer  
    toc
        tic
    for ip_ch = 1 : cache.ip_channel
        for oh = 1 : output_h
            for ow = 1 : output_w
             tmp = delta(((oh-1)+1):((oh-1)+cache.W_height),...
                         ((ow-1)+1):((ow-1)+cache.W_width),:,:);
             dx(oh,ow,ip_ch,:) = sum(sum(sum(tmp .* rot90(rot90(W(:,:,:,ip_ch))))));
            end
        end           
    end             %calculate dx to previous layer  
    toc
    if cache.flag=="same"
        [x_ap_height,x_ap_width,~,~]=size(cache.x_ap);
        left_cut=floor((x_ap_width-ip_width)/2);
        right_cut=ceil((x_ap_width-ip_width)/2);
        up_cut=floor((x_ap_height-ip_height)/2);
        down_cut=ceil((x_ap_height-ip_height)/2);
        dx=dx(up_cut+1:x_ap_height-down_cut,left_cut+1:x_ap_width-right_cut,:,:);
    end
    
    %expand gradient to the result of which use Stribe=1 to cov input   
    grad = delta(padding_h+1:S_1_dY_h-padding_h,padding_w+1:S_1_dY_w-padding_w,:,:);
%     grad=zeros(S_1_dY_h-2*patchh,S_1_dY_w-2*patchl,dx_deep);
%     for i=1:1:length(dx(1,1,:))
%         for j=1:1:length(dx(1,:,1))
%             for k=1:1:length(dx(:,1,1))
%                 grad(k*stridel-(stridel-1),...
%                 j*strideh-((strideh-1)),...
%                 i)=dx(k,j,i);
%             end
%         end
%     end             %expand gradient to the result of which use Stribe=1 to cov input    

    [gradh,gradew,~,~]=size(grad);
    dw=zeros(size(cache.W));

% calculate dw 
    for ip_ch = 1 : cache.ip_channel
        for oh = 1 : cache.W_height
            for ow = 1 : cache.W_width
                 tmp0 = cache.x_ap(((oh-1)+1):((oh-1)+gradh),...
                                  ((ow-1)+1):((ow-1)+gradew),...
                                    ip_ch,:);
                 tmp1 = sum(sum(tmp0 .* grad(:,:,:,:)));
                 dw(oh,ow,ip_ch,:) = sum(permute(tmp1,[1 2 4 3]))./batch;
%                  dw(oh,ow,:,ip_ch) = permute(tmp1,[1 2 4 3]);
            end
        end 
    end
              
%     dw = permute(dw,[1 2 4 3]);
    db = permute(dY,[1 2 4 3]);
    db = sum(sum(sum(db)))./batch;   %calculate db  
    db = permute(db,[3 4 1 2]);
end