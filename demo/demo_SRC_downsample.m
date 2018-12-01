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
% given downsample factors
s = [9 10 12 18 20 24 30 40 60];
% experiment results: error and runtime
[err_CMU_down, t_CMU_down] = run_SRC_downsample(TrainSet, TestSet, [1],120,128);




%% SRC with eigenfaces on Yale database
fprintf('running experiment\n\t SRC with eigenfaces on Yale database...\n')
load('Yale_face_full.mat'); 
% set up dataset structure
TrainSet = {};
TrainSet.X = YaleTrain;
TrainSet.y = classTrain;
TestSet = {};
TestSet.X = YaleTest;
TestSet.y = classTest;
% given downsample factors
s = [6 7 13	15 18 25 28 33 40 57 80];
% experiment results: error and runtime
[err_Yale_eigen, t_Yale_eigen] = run_SRC_downsample(TrainSet, TestSet, s, 192,168);



%% experiment on AR database
fprintf('running experiment\n\t SRC with eigenfaces on AR database...\n')
load('AR_Face_img_60x43.mat'); 
% given experiment feature dimensions
s   = [5 10 20 30 40 50 100 150 200 train_num];
% experiment results: error and runtime
[err_AR_eigen, t_AR_eigen] = run_SRC_downsample(TrainSet, TestSet, s);
