close all;
clear;
clc;


%% SRC with eigenfaces on CMU database
fprintf('running experiment\n\t SRC with eigenfaces on CMU database...')
load('CMU_Dataset_Test+Train.mat')
% given experiment feature dimensions
s   = [5 10 20 30 40 50 100 150 200 train_num];
% experiment results: error and runtime
[err_CMU_eigen, t_CMU_eigen] = run_SRC_eigen(TrainSet, TestSet, s);




%% SRC with eigenfaces on Yale database
fprintf('running experiment\n\t SRC with eigenfaces on Yale database...')
load('Yale_face_full.mat'); 
% set up dataset structure
TrainSet = {};
TrainSet.X = YaleTrain;
TrainSet.y = classTrain;
TestSet = {};
TestSet.X = YaleTest;
TestSet.y = classTest;
% experiment feature dimensions
s   = [5 10 20 30 40 50 100 150 200 length(TrainSet.y)];
% experiment results: error and runtime
[err_Yale_eigen, t_Yale_eigen] = run_SRC_eigen(TrainSet, TestSet, s);



%% experiment on AR database
fprintf('running experiment\n\t SRC with eigenfaces on AR database...')
load('AR_Face_img_60x43.mat'); 
% given experiment feature dimensions
s   = [5 10 20 30 40 50 100 150 200 train_num];
% experiment results: error and runtime
[err_AR_eigen, t_AR_eigen] = run_SRC_eigen(TrainSet, TestSet, s);
