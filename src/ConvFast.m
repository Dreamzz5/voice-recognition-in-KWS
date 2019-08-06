function [y,cache] = ConvFast(Input,W,b,stride_h,stride_w,flag)
    [W_height,W_width,Input_channel,op_channel] = size(W);
    [ip_height,ip_width,ip_channel,batch] = size(Input);
    x_height = ip_height;
    x_width = ip_width;
    x = Input;

    x_ap = x;  %x_after_padding
%     b = permute(b,[3 1 2 4]);
    if(ip_channel ~= Input_channel)
        fprintf("error W matrix!");
        y = -1;
    end
    
    if(flag == "valid")
        MOD_h = mod((x_height - W_height),stride_h);
        MOD_w = mod((x_width - W_width),stride_w);
        if(MOD_h==0)
           x = x;
        else
           for count = 1:MOD_h
               x(x_height-(count-1),:,:,:) = [];
           end
        end
        if(MOD_w==0)
           x = x;
        else
           for count = 1:MOD_w
               x(:,x_width-(count-1),:,:) = [];
           end
        end
        output_h = floor((x_height - W_height)/stride_h + 1);
        output_w = floor((x_width - W_width)/stride_w + 1);        
        %%==================================================================  
        %   x_new:
        %       ip_channel-1    ip_channel-2  ...  ip_channel-ip_channel
        %     [             ]   [            ]     [                 ]
        %     [             ]   [            ]     [                 ]            
        %     ...
        %     [             ]   [            ]     [                 ]
        %%==================================================================            
        xnew_h = output_h * output_w;
        xnew_w = W_height * W_width;
        xnew = zeros(xnew_h,xnew_w*ip_channel,batch);
        for xipch = 1 : ip_channel
            for xh = 1:xnew_h
                xnew(xh,(xipch-1)*xnew_w+1:(xipch-1)*xnew_w+xnew_w,:) = ...
                    reshape(x((stride_h*floor((xh-1)/output_w)+1):(stride_h*floor((xh-1)/output_w)+W_height),...
                         (stride_w*mod(xh-1,output_w)+1):(stride_w*mod(xh-1,output_w)+W_width),...
                          xipch,:),1,xnew_w,1,batch);
            end
        end
        %%=======================================================================  
        %   W_new:
        %        op_channel-1         op_channel-2             op_channel-end
        %     [  ip_channel-1  ]   [  ip_channel-1  ]  ...  [  ip_channel-1  ]  
        %     ...                  ...                      ...
        %     [                ]   [                ]       [                ]
        %     [  ip_channel-2  ]   [  ip_channel-2  ]       [  ip_channel-2  ]
        %     ...                  ...                      ...
        %     [                ]   [                ]       [                ]
        %     [ ip_channel-end ]   [ ip_channel-end ]       [ ip_channel-end ]
        %     ...                  ...                      ...
        %     [                ]   [                ]       [                ]
        %%=======================================================================          
        Wnew_h = W_height * W_width;
        Wnew_w = op_channel;
        Wnew = zeros(Wnew_h*ip_channel,Wnew_w);
        for wipch = 1:ip_channel
            for Wopch = 1:Wnew_w
                Wnew((wipch-1)*Wnew_h+1:(wipch-1)*Wnew_h+Wnew_h,Wopch) = ...
                    reshape(W(:,:,wipch,Wopch),Wnew_h,1);
            end
        end             
        %%================================================================== 
        t1 = tic;
        ynew = mtimesx(xnew,Wnew) + b;
        y = permute(reshape(ynew,output_w,output_h,op_channel,batch),[2 1 3 4]);
        mtimes = toc(t1);
        ynew2 = zeros(xnew_h,Wnew_w,batch);
%         t2 = tic;
%         for i = 1:batch
%             ynew2(:,:,i) = xnew(:,:,i) * Wnew;
%         end
%         ynew2 = ynew2 + b;
%         y2 = permute(reshape(ynew2,output_w,output_h,op_channel,batch),[2 1 3 4]);
%         fortimes = toc(t2)
        %%==================================================================  
    elseif(flag == "same")
        padding_h = x_height*(stride_h - 1) + W_height - stride_h;
        padding_w = x_width*(stride_w - 1) + W_width - stride_w;
        tmp = zeros(x_height+padding_h,x_width,ip_channel,batch);
        if mod(padding_h,2) == 0
           tmp((1+padding_h/2):(x_height+padding_h/2),:,:,:) = x;
        else
           tmp((1+(padding_h-1)/2):(x_height+(padding_h-1)/2),:,:,:) = x;
        end
        x = tmp;
        [x_height,x_width,x_channel,batch] = size(x);
        tmp = zeros(x_height,x_width+padding_w,x_channel,batch);
        if mod(padding_w,2) == 0
           tmp(:,(1+padding_w/2):(x_width+padding_w/2),:,:) = x;
        else
           tmp(:,(1+(padding_w-1)/2):(x_width+(padding_w-1)/2),:,:) = x;
        end
        x = tmp;
        x_ap = x;  %x_after_padding
        [x_height,x_width,x_channel,batch] = size(x);
        output_h = floor((x_height - W_height)/stride_h + 1);
        output_w = floor((x_width - W_width)/stride_w + 1);
        %%==================================================================  
        %   x_new:
        %       ip_channel-1    ip_channel-2  ...  ip_channel-ip_channel
        %     [             ]   [            ]     [                 ]
        %     [             ]   [            ]     [                 ]            
        %     ...
        %     [             ]   [            ]     [                 ]
        %%==================================================================            
        xnew_h = output_h * output_w;
        xnew_w = W_height * W_width;
        xnew = zeros(xnew_h,xnew_w*ip_channel,batch);
        for xipch = 1 : ip_channel
            for xh = 1:xnew_h
                xnew(xh,(xipch-1)*xnew_w+1:(xipch-1)*xnew_w+xnew_w,:) = ...
                    reshape(x((stride_h*floor((xh-1)/output_w)+1):(stride_h*floor((xh-1)/output_w)+W_height),...
                         (stride_w*mod(xh-1,output_w)+1):(stride_w*mod(xh-1,output_w)+W_width),...
                          xipch,:),1,xnew_w,1,batch);
            end
        end
        %%=======================================================================  
        %   W_new:
        %        op_channel-1         op_channel-2             op_channel-end
        %     [  ip_channel-1  ]   [  ip_channel-1  ]  ...  [  ip_channel-1  ]  
        %     ...                  ...                      ...
        %     [                ]   [                ]       [                ]
        %     [  ip_channel-2  ]   [  ip_channel-2  ]       [  ip_channel-2  ]
        %     ...                  ...                      ...
        %     [                ]   [                ]       [                ]
        %     [ ip_channel-end ]   [ ip_channel-end ]       [ ip_channel-end ]
        %     ...                  ...                      ...
        %     [                ]   [                ]       [                ]
        %%=======================================================================          
        Wnew_h = W_height * W_width;
        Wnew_w = op_channel;
        Wnew = zeros(Wnew_h*ip_channel,Wnew_w);
        for wipch = 1:ip_channel
            for Wopch = 1:Wnew_w
                Wnew((wipch-1)*Wnew_h+1:(wipch-1)*Wnew_h+Wnew_h,Wopch) = ...
                    reshape(W(:,:,wipch,Wopch),Wnew_h,1);
            end
        end             
%%==================================================================     
        ynew = mtimesx(xnew,Wnew) + b;
        y = permute(reshape(ynew,output_w,output_h,op_channel,batch),[2 1 3 4]);   
    else
        fprintf('please input the flag');
        y = -2;
    end

    cache = struct('Input',Input,'W',W,'stride_h',stride_h,...
              'stride_w',stride_w,'MOD_h',MOD_h,'MOD_w',MOD_w,...
              'W_height',W_height,'W_width',W_width,...
              'ip_channel',ip_channel,'op_channel',op_channel,...
              'ip_height',ip_height,'ip_width',ip_width,...
              'flag',flag,'x_ap',x_ap);
end