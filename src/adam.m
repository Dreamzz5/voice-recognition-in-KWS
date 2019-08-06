function [out,GradientMovingAverage,SquaredGradientMovingAverage] = adam(globalLearnRate,this,LocalL2Factors,GradientMovingAverage,SquaredGradientMovingAverage,t)

beta1 = 0.9;
beta2 = 0.999;
epsilon =  1e-8;
normThreshold = 10;

globalL2Factor = 1e-4;
for i = 1:length(this)  
  effectiveL2Factor = LocalL2Factors(i)*globalL2Factor;
  gradients{i} = effectiveL2Factor.*this(i).forward + this(i).back;
end


gradients = iThresholdL2Norm(gradients,normThreshold);%% tiducaijian

learnRateShrinkFactor = sqrt(1-beta2^t)/(1-beta1^t);
effectiveLearningRate = learnRateShrinkFactor.*globalLearnRate.*1;
for i = 1:length(gradients)
  GradientMovingAverage{i} = beta1.*GradientMovingAverage{i} + (1 - beta1).*gradients{i};
  SquaredGradientMovingAverage{i} = beta2.*SquaredGradientMovingAverage{i} + (1 - beta2).*(gradients{i}.^2);
  out{i} = -effectiveLearningRate.*(GradientMovingAverage{i}./(sqrt(SquaredGradientMovingAverage{i}) + epsilon) );
end
end


function originalGrads = iThresholdL2Norm(originalGrads, normThreshold)
for i = 1:numel(originalGrads)
    squareSum = sum(originalGrads{i}(:).^2);
    gradNorm = sqrt(squareSum);
    if gradNorm > normThreshold
%         disp('²Ã¼ô');
        originalGrads{i} = originalGrads{i}*(normThreshold/gradNorm);
    end
end
end

