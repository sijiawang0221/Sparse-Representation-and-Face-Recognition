close all;
clear;
clc;
load('Yale_Face_Full.mat')
%%
TestSet.X = YaleTest;
% TestSet.X = CMUTest;
% TestSet.y = classTest;
TrainSet.X = YaleTrain;
% TrainSet.X = CMUTrain;
% TrainSet.y = classTrain;

%Set up class vectors from data that was read in
TestSet.y = classTest;
TrainSet.y = classTrain;
%Set up initial parameters: number of classes, number of test images,
%number of train images
class_num = 38;
test_num = 1209;
train_num = 1205;
%set up parameters for downsampling: size of each image (n x m),
%downsampling factor
% N = 192;
% M = 168;
% Downsample_Factor = 1/2;
% temp = reshape(YaleTest(:,1),[N M]);
% temp2 = imresize(temp,Downsample_Factor);
% TestSet.X = zeros(length(temp2(:)),test_num);
% [~, L] = size(YaleTest);
% for i=1:L
%     temp = reshape(YaleTest(:,i),[N M]);
%     temp2 = imresize(temp,Downsample_Factor);
%     TestSet.X(:,i) = temp2(:);
% end
% temp = reshape(YaleTrain(:,1),[N M]);
% temp2 = imresize(temp,Downsample_Factor);
% TrainSet.X = zeros(length(temp2(:)),train_num);
% [~, L] = size(YaleTrain);
% for i=1:L
%     temp = reshape(YaleTrain(:,i),[N M]);
%     temp2 = imresize(temp,Downsample_Factor);
%     TrainSet.X(:,i) = temp2(:);
% end


%% reduce dataset for a quick test if necessary
% max_class_num = 20;
% max_train_samples = 309;
% max_test_samples = 315;
% [TrainSet, TestSet, train_num, test_num, class_num] = reduce_dataset(TrainSet, TestSet, max_class_num, max_train_samples, max_test_samples);


%% set paramters
eigenface_flag = true;
% eigenface_dim = floor(train_num/2); % example
% dim = size(TrainSet.X, 1);
% if eigenface_dim > dim
%     eigenface_dim = dim;
% end


%% normalize dataset
[TrainSet_normalized.X, TrainSet_normalized.y] = data_normalization(TrainSet.X, TrainSet.y, 'std');        
[TestSet_normalized.X, TestSet_normalized.y] = data_normalization(TestSet.X, TestSet.y, 'std');     


%% SVM
options.verbose = true;
% options.eigenface = eigenface_flag;
% options.eigenface_dim = 5; %eigenface_dim;
tic;
accuracy_svm = svm_classifier(TrainSet, TestSet, train_num, test_num, class_num, options);
timing_svm = toc;
fprintf('# SVM: Accuracy = %5.5f\n', accuracy_svm);


%% LSR
lambda = 0.001;
options.verbose = true;
% options.eigenface_dim = 5; %eigenface_dim;
tic;
[accuracy_lsr, ~, ~] = lsr(TrainSet, TestSet, train_num, test_num, class_num, lambda, options);
timing_lsr = toc;
fprintf('# LSR: Accuracy = %5.5f\n', accuracy_lsr);


%% LRC
% clear options;
options.verbose = true;
% options.eigenface_dim = 5; %eigenface_dim;
tic;
accuracy_lrc = lrc(TrainSet_normalized, TestSet_normalized, test_num, class_num, options);
timing_lrc = toc;
fprintf('# LRC: Accuracy = %5.5f\n', accuracy_lrc);


% %% LDRC
% clear options;
% options.verbose = true;
% accuracy_ldrc = ldrc(TrainSet_normalized, TestSet_normalized, train_num, test_num, class_num, eigenface_dim, options);
% fprintf('# LDRC: Accuracy = %5.5f\n', accuracy_ldrc);
% 
% 
% %% LCDRC
% clear options;
% options.verbose = true;
% accuracy_lcdrc = lcdrc(TrainSet_normalized, TestSet_normalized, train_num, test_num, class_num, eigenface_dim, options);
% fprintf('# LCDRC: Accuracy = %5.5f\n', accuracy_lcdrc);





%% display accuracy
fprintf('\n\n## Summary of results\n\n')
fprintf('# SVM: Accuracy = %5.5f\n', accuracy_svm);
fprintf('# LSR: Accuracy = %5.5f\n', accuracy_lsr);
fprintf('# LRC: Accuracy = %5.5f\n', accuracy_lrc);
fprintf('# SVM: Timing = %5.5f\n', timing_svm);
fprintf('# LSR: Timing = %5.5f\n', timing_lsr);
fprintf('# LRC: Timing = %5.5f\n', timing_lrc);
% fprintf('# LDRC: Accuracy = %5.5f\n', accuracy_ldrc);
% fprintf('# LCDRC: Accuracy = %5.5f\n', accuracy_lcdrc);
