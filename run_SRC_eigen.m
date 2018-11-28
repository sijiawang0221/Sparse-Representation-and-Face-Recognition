function [err, t] = run_SRC_eigen(TrainSet, TestSet, feature_dims)
% This fuction will yields result of SRC+eigenface given training and test
% dataset.
% 
% Input: 
%       TrainSet.X      training samples of size m*n
%       TrainSet.y      training labels for training samples    
%       TestSet.X       test samples of size m*n
%       TesrSet.y       test labels for training samples 
% Output:
%       err             a vector of error for each feature dimensions
%       t               computation time of the experiment

% given experiment feature dimensions
num = length(s);

% t is the time last of each experiment 
% err is the prediction error 
t   = zeros(1,num);
err = zeros(1,num);

for i = feature_dims
    fprintf('running %d th experiment with %d feature dimension\n', i, s(i));
    % set eigenface parameter
    options.eigenface     = true;
    options.eigenface_dim = feature_dims(i);
    
    % run experiment
    tic()
    [label, ~] = SRC(TrainSet, TestSet, 0.05, 0.001, options);
    t(i)   = toc();
    err(i) = sum(label == TestSet.y)/length(TestSet.y);
end