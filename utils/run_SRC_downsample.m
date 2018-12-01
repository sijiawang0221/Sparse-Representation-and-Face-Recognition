function [err, t] = run_SRC_downsample(TrainSet, TestSet, downsample_factors,W,H)
% This fuction will yields result of SRC+eigenface given training and test
% dataset.
% 
% Input: 
%       TrainSet.X          training samples of size m*n
%       TrainSet.y          training labels for training samples    
%       TestSet.X           test samples of size m*n
%       TesrSet.y           test labels for training samples 
%       downsample_factors  a vector indicate the down sample factors
% 
% Output:
%       err             a vector of error for each feature dimensions
%       t               computation time of the experiment



    % given experiment feature dimensions
    num = length(downsample_factors);

    % t is the time last of each experiment 
    % err is the prediction error 
    t   = zeros(1,num);
    err = zeros(1,num);

    for j = 1:num
        fprintf('running %d th experiment with 1/%d feature dimension\n', j, downsample_factors(j));
        % set eigenface parameter
        options.eigenface     = false;

        Train_down = downSample(TrainSet, W, H, downsample_factors(j));
        Test_down  = downSample(TestSet,  W, H, downsample_factors(j));
        % run experiment
        tic()
        [label, ~] = SRC(Train_down, Test_down, 0.05, 0.001, options);
        t(j)   = toc();
        err(j) = sum(label == Test_down.y)/length(Test_down.y);

    end
    
end