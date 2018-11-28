close all;
clear;
clc;
load('CMU_Dataset_New.mat')

%
%%Set up class vectors from data that was read in
TestSet.y = classTest;
TrainSet.y = classTrain;
%Set up initial parameters: number of classes, number of test images,
%number of train images
class_num = 20;
test_num = 311;
train_num = 313;
%set up parameters for downsampling: size of each image (n x m),
%downsampling factor
N = 120;
M = 128;
Downsample_Factor = [1,1/2,1/3,1/4,1/5,1/6]; %choose value less than 1

%perform downsampling to get output dimensions
temp = reshape(CMUTest(:,1),[N M]);
temp2 = imresize(temp,Downsample_Factor);
%preallocate TestSet.X
for i = Downsample_Factor
    TestSet.X = zeros(length(temp2(:)),test_num);
    [~, L] = size(CMUTest);
    for i=1:L
        temp = reshape(CMUTest(:,i),[N M]);
        temp2 = imresize(temp,Downsample_Factor);
        TestSet.X(:,i) = temp2(:);
    end

    %perform downsampling to get output dimensions
    temp = reshape(CMUTrain(:,1),[N M]);
    temp2 = imresize(temp,Downsample_Factor);
    %preallocate TrainSet.X
    TrainSet.X = zeros(length(temp2(:)),train_num);
    [~, L] = size(CMUTrain);
    for i=1:L
        temp = reshape(CMUTrain(:,i),[N M]);
        temp2 = imresize(temp,Downsample_Factor);
        TrainSet.X(:,i) = temp2(:);
    end
end