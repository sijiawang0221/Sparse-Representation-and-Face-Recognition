function [err, t] = run_SRC_fisher(TrainSet, TestSet, fisher_dim, Wopt)
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
    num = length(fisher_dim);

    % t is the time last of each experiment 
    % err is the prediction error 
    t   = zeros(1,num);
    err = zeros(1,num);

    for j = 1:num
        fprintf('running %d th experiment with 1/%d feature dimension\n', j, fisher_dim(j));
        % set eigenface parameter
        options.eigenface     = false;
        
        Train_fisher = {};
        Train_fisher.y = TrainSet.y;
        Train_fisher.X = Wopt(:,1:fisher_dim(j))' * TrainSet.X;
        Test_fisher = {};
        Test_fisher.X = Wopt(:,1:fisher_dim(j))' * TestSet.X;
        Test_fisher.y = TestSet.y;

        % run experiment
        tic()
        [label, ~] = SRC(Train_fisher, Test_fisher, 0.05, 0.001, options);
        t(j)   = toc();
        err(j) = sum(label == Test_fisher.y)/length(Test_fisher.y);

    end
    
end