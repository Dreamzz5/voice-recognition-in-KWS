function [y,cache] = Conv2D_forward(Input,W,b,stride_h,stride_w,flag)
    [W_height,W_width,Input_channel,op_channel] = size(W);
    [ip_height,ip_width,ip_channel,batch] = size(Input);
    x_height = ip_height;
    x_width = ip_width;
    x = Input;
    x_ap = x;  %x_after_padding
    b = permute(b,[3 1 2 4]);
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
        y = zeros(output_h,output_w,op_channel,batch);
        for op_ch = 1 : op_channel
           %for ip_ch = 1 : ip_channel
              for oh = 1 : output_h
                 for ow = 1 : output_w
                     tmp0 = x((stride_h*(oh-1)+1):(stride_h*(oh-1)+W_height),...
                             (stride_w*(ow-1)+1):(stride_w*(ow-1)+W_width),...
                              :,:);
                     y(oh,ow,op_ch,:) = sum(sum(sum(tmp0 .* W(:,:,:,op_ch))));
                 end
              end
              %y(:,:,op_ch,:) = y(:,:,op_ch,:) + tmp1(:,:,ip_ch,:);
          % end
        end 
        y = y + b;
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
        y = zeros(output_h,output_w,op_channel,batch);
        tmp = [];
        for op_ch = 1 : op_channel
           for ip_ch = 1 : Input_channel
              for oh = 1 : output_h
                 for ow = 1 : output_w
                    tmp0 = x((stride_h*(oh-1)+1):(stride_h*(oh-1)+W_height),...
                             (stride_w*(ow-1)+1):(stride_w*(ow-1)+W_width),...
                              ip_ch,:);
                    tmp1(oh,ow,ip_ch,:) = sum(sum(tmp0 .* W(:,:,ip_ch,op_ch)));
        %                     for wh = 1: W_hight
        %                         for ww = 1: W_width
        %                             y(oh,ow,op_ch) = y(oh,ow,op_ch)...
        %                                 + x(oh+wh-1,ow+ww-1) * W(wh,ww,ip_ch,op_ch);
        %                         end
        %                     end
        %                     if ip_ch == Input_channel
        %                         y(oh,ow,op_ch) = y(oh,ow,op_ch) + b(op_ch);
        %                     end
                 end
              end
              y(:,:,op_ch,:) = y(:,:,op_ch,:) + tmp1(:,:,ip_ch,:);
           end
        end 
        y = y + b;
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