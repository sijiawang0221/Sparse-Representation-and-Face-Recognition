close all;
clear;
clc;

% load dataset
load('AR_Face_img_60x43.mat'); 

% given experiment feature dimensions
s   = [5 10 20 30 40 50 150 200 train_num];
num = length(s);

% t is the time last of each experiment 
% err is the prediction error 
t   = zeros(1,num);
err = zeros(1,num);


for i = 1:num
    % set eigenface parameter
    options.eigenface     = true;
    options.eigenface_dim = s(i);
    
    % run experiment
    tic()
    [label, iterationCount] = SRC(TrainSet, TestSet, 0.05, 0.001, options);
    t(i)   = toc();
    err(i) = sum(label == TestSet.y)/length(TestSet.y);
end