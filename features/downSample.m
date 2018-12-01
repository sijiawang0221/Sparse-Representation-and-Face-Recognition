function [Data_downsampled] = downSample(DataSet, W, H, Downsample_Factor)
% This fuction downsample the original dataset to lower resolution
% Input: 
%       DataSet             Original dataset
%       W                   Width of the original image                
%       H                   Height of the original image    
%       Downsample_factor   Downsample factor
% Output:
%       Data_downsampled    Downsampled dataset

    % get the number of samples
    num = length(DataSet.y);
    W2 = round(W/Downsample_Factor);
    H2 = round(H/Downsample_Factor);
    
    Data_downsampled.X = zeros(W2*H2,num);
    Data_downsampled.y = DataSet.y;

    for i=1:num
        temp = reshape(DataSet.X(:,i),[W H]);
        temp2 = imresize(temp,[W2 H2]);
        Data_downsampled.X(:,i) = temp2(:);
    end
end