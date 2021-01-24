% clc; clear all; close all;
%% Main function-calls functions that compute weights of the network and outputs optimal weights
function Weights = MyPTA()

[TrainImg, TrainLabel, ~, ~] = HW2_MNIST(60000, 10);
Vectorized_train = VectorizeMx(TrainImg);
% Vectorized_test = VectorizeMx(TestImg);

imageNumber = length(TrainImg);

Weights = PerceptronAlg(1, 1e-6, imageNumber, Vectorized_train, TrainLabel);
end

%% This function vectorizes 28*28 input images
function Vectorized = VectorizeMx(Mx)
    Vectorized = zeros(784, length(Mx));
    for j=1:length(Mx)
        Vectorized(:,j) = reshape(Mx{j}',[],1);
    end
end

%% Activation function : in this question, we used step function
function Activ_Out = ActivationStepfunction(vect) % gets a vector :W*x(i)
    Activ_Out = vect;
    Activ_Out(Activ_Out>0)= 1; Activ_Out(Activ_Out<0)= 0;
end

%% This function generates "d" vectores
function dx_i = DesireOutput(ActualLabel, len)
    dx_i = zeros(len, 1);
    dx_i(ActualLabel+1) = 1;
end

%% "x" is the data (can be train set or test set) -- "N" is the number of images in the dataset
function W = PerceptronAlg(eta, tolerance, N, x, LabeL) 
iter=0;
%  imagesc(reshape(x(:,1), 28, 28)'); colormap(gray(256));
    W = rand(10, 784); W(W < 1e-7) = 0;
%     W = randi([0 1], 10, 784);  W(W < 1e-7) = 0;
    epoch = 1; errors = zeros(1,1000);
    errors(epoch) = 0;
    LabeL = LabeL';
    while iter <1000
        iter = iter+1
        %% Compute "induced local field" for all images at once:
        v = W*x; %(:,i);
        %% Get the max value of "v" at each column (corresponding to each image):
        %% STEP 3.1.1.2)
        Largest_v = max(v);
        [~, indexLargest_v] = max(v, [], 1); 
        indexLargest_v = indexLargest_v - 1; %% Matlab indexing scheme (YEEEES :) )
        for i=1:size(x,2)
            % Counts misclassification error
            if indexLargest_v(i) ~= LabeL(i), errors(epoch) = errors(epoch) + 1; end    
        end
        errors(epoch)
%         myDebug = [indexLargest_v; LabeL];
        %% STEP 3.1.2)
        epoch = epoch +1;
        %%STEP 3.1.3)
        %% STEP 3.1.3.1) updating the weights here:
        for i=1:N
            d_xi = DesireOutput(LabeL(i), 10); % Desire output per (training) image
            u = ActivationStepfunction(v(:,i)); % Activation function per (training) image
            subtrct = (d_xi - u);
            penalize = subtrct * x(:,i)'; %%eta* 
            W = W + penalize;
        end 
       if errors(epoch-1)/N < tolerance
           plot(1:epoch-1,errors(1:epoch-1),'red','linewidth',2); hold on
           scatter(1:epoch-1,errors(1:epoch-1),'red','filled')
           xlabel('Epoch'); ylabel('Misclassification error');
           title(['Error plot -- n = ' num2str(N) ' (images), \eta = ' num2str(eta) ', \epsilon = ' num2str(tolerance)])
           return
       end
       
    end
end