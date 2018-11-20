close all;
clear;
clc;

% load dataset
load('CMU_Dataset_Test+Train.mat'); 

% given experiment feature dimensions
s   = [5 10 20 30 40 50 150 200 length(TrainSet.y)];
num = length(s);

% t is the time last of each experiment 
% err is the prediction error 
t   = zeros(1,num);
err = zeros(1,num);


for i = 1:3
    fprintf('running %d th experiment with %d feature dimension\n', i, s(i));
    % set eigenface parameter
    options.eigenface     = true;
    options.eigenface_dim = s(i);
    
    % run experiment
    tic()
    [label, iterationCount] = SRC(TrainSet, TestSet, 0.05, 0.001, options);
    t(i)   = toc();
    err(i) = sum(label == TestSet.y)/length(TestSet.y);
end