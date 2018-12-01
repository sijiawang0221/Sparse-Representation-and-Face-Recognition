% close all;
% clear;
% clc;
cd ..;

%% SRC with eigenfaces on CMU database
fprintf('running experiment\n\t SRC with eigenfaces on CMU database...\n')
load('CMU_Dataset_new.mat')
% set up dataset structure
TrainSet = {};
TrainSet.X = CMUTrain;
TrainSet.y = classTrain;
TestSet = {};
TestSet.X = CMUTest;
TestSet.y = classTest;
% given experiment feature dimensions
s   = [19 10 5];
% experiment results: error and runtime
[err_CMU_fisher, t_CMU_fisher] = run_SRC_fisher(TrainSet, TestSet, s, Wopt);


%% SRC with eigenfaces on Yale database
fprintf('running experiment\n\t SRC with eigenfaces on Yale database...\n')
load('datasets/Yale_face_full.mat'); 
% set up dataset structure
TrainSet   = {};
TrainSet.X = YaleTrain;
TrainSet.y = classTrain;
TestSet    = {};
TestSet.X  = YaleTest;
TestSet.y  = classTest;
% given downsample factors
s = [length(unique(TrainSet.y))-1	30	20	10	5];
% experiment results: error and runtime
[err_Yale_fisher, t_Yale_fisher] = run_SRC_fisher(TrainSet, TestSet, s, Wopt);


%% experiment on AR database
fprintf('running experiment\n\t SRC with eigenfaces on AR database...\n')
load('AR_Face_img_60x43.mat'); 
% given experiment feature dimensions
s   = [5 10 20 30 40 50 100 150 200 train_num];
% experiment results: error and runtime
[err_AR_down, t_AR_down] = run_SRC_downsample(TrainSet, TestSet, s);
