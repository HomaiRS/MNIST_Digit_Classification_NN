function errors = myPTA_OnTestSet()
    [~, ~, TestImg, TestLabel] = HW2_MNIST(10,10000)
    Weights = MyPTA(); % Computed Weights on the train set
    TestImg_Vect = VectorizeMx(TestImg) %% x(i)
    
    N = length(TestImg); errors =0;
    TestLabel = TestLabel'; 
    %% Compute "induced local field" for all images at once:
    v = Weights*TestImg_Vect; %(:,i);
    %% Get the max value of "v" at each column (corresponding to each image):
    %% STEP 3.1.1.2)
    Largest_v = max(v);
    [~, indexLargest_v] = max(v, [], 1); 
    indexLargest_v = indexLargest_v - 1; %% Matlab indexing scheme (YEEEES :) )
    myDebug = [indexLargest_v; TestLabel];
%     for i=1:size(TestImg,2)
%         % Counts misclassification error
%         if indexLargest_v(i) ~= TestLabel(i), errors = errors + 1; end    
%     end
    %% My robust version -- avoiding above for loop (unnecessary)
    errors = length(find(indexLargest_v ~= TestLabel))
    
end

%% This function vectorizes 28*28 input images
function Vectorized = VectorizeMx(Mx)
    Vectorized = zeros(784, lengt1h(Mx));
    for j=1:length(Mx)
        Vectorized(:,j) = reshape(Mx{j}',[],1);
    end
end