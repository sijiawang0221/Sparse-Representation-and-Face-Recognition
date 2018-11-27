%INPUT: 
% X | double | Nxn | data matrix: rows->observations cols->variables 
% label | double | Nx1 | labels 
% OUTPUT: 
% Y | double | nxm | reduced data 
% W | double | kxm | transformationmatrix Y = X*W 
function [ Y, Wopt] = fisherLDA( X, label)

classes = unique(label); 
k = numel(classes); % number of classes 
dim = size(X,2); % Dimension of input data

SB = zeros(dim); % between-class scatter matrix 
SW = zeros(dim); % within-class scatter matrix

X_ = mean(X); % mean of complete data

% loop over all classes and calculate SW and SB 
for i = 1 : k 
v = label == classes(i); 
Xl = X(v,:); % all datapoints corresponding to label i 

Xl_ = mean(Xl); % mean of all datapoints corresponding to label i 

r = Xl_-X_; 
SB = SB + size(Xl,1)*( r'*r ); 

for j = 1 : size(Xl,1) 
r = Xl(j,:)-Xl_; 
SW = SW + r'*r; 
end 

end 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Note finding SW and SB is expensive!        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%load('CMU_Fisher_SWSBW.mat')

%Calculate Wpca and Wfld
Wpca = pca(X);
N = size(X,1);
reduce_dim = N - k; %N-c, Images - classes
Wpca = Wpca(:,1:reduce_dim); %n by (N-c)
SB_reduced = Wpca.' * SB * Wpca;
SW_reduced = Wpca.' * SW * Wpca;
%Solve generalized eigenvalue problem
%SB*w = lambda*SW*w
[Wfld, LAMBDA] = eig(SB_reduced, SW_reduced);
lambda=diag(LAMBDA);
[~, SortOrder] = sort(lambda,'descend');
Wfld = Wfld(:,SortOrder);
m = k -1; %Note there are at most c-1 nonzero eigvals
Wfld = Wfld(:,1:m); %(N-c) by m

%Calculate the optimal Projection Operator Wopt
Wopt = Wpca * Wfld; %n by m
%Normalize Wopt
% for i=1:length(m)
%     v = Wopt(:,i);
%     Wopt(:,i) = (v-min(v))/(max(v)-min(v));
% end
Y = Wopt.' * X.';

end