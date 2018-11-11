function [label, iterationCount] = SRC(tr, te, epsilon, lambda, options)
% This fuction implement SRC algorithm
% Input: 
%       tr.x        training samples of size m*n
%       tr.y        class label of training 
%       te          a test sample of size m*1    
%       epsilon     optimal error tolerance epsilon
%       lambda      coefficient controls data fidelity
% Output:
%       x           sparse representation, x \in R^{n*1}
% Reference
%       [1] J. Wright, et al. Robust Face Recognition via Sparse Representation
%       [2] https://github.com/hiroyuki-kasai/ClassifierToolbox
%       [3] L. Zhang, et all. A Simple Homotopy Algorithm for Compressive Sensing

% get the size of training sample
[m, n] = size(tr.x);

% extract options    
if ~isfield(options, 'eigenface')
    eigenface = true;
else
    eigenface = options.eigenface;
end    

if ~isfield(options, 'eigenface_dim')
    eigenface_dim = n;
else
    eigenface_dim = options.eigenface_dim;
end

% generate eigenface
if eigenface    
    [disc_set, ~, ~] = Eigenface_f(tr.x, eigenface_dim);
    tr.x  =  disc_set' * tr.x;
    te.x  =  disc_set' * te.X;
end

% normalize the columns of A to have unit l2-norm
tr.x = tr.x ./ sqrt(sum(tr.x .^ 2, 2));
te.x = te.x ./ sqrt(sum(te.x .^ 2, 2));

% get the class label
classes = unique(tr.y);
class_num = length(classes);

% solve the l1-minimization problem
P = 1/(tr.x'*tr.x+1e-5*eye(n))*tr.x';
x0 = P*te.x;


% implement Homotopy algorithm
maxIteration = 5000;
isNonnegative = false;
stoppingCriterion = -1;
[xp, iterationCount] = SolveHomotopy(tr.x, te.x, ...
                        'maxIteration', maxIteration,...
                        'isNonnegative', isNonnegative, ...
                        'stoppingCriterion', stoppingCriterion, ...
                        'groundtruth', x0, ...
                        'lambda', lambda, ...
                        'tolerance', epsilon);  


% calculate residual for each class
residuals = zeros(1, max(classes));
for j = 1 :class_num
    idx = find(tr.y == classes(j));
    residuals(int(classes(j))) = norm(te.x - tr.x(:,idx)*xp(idx));
end

% get the predicted label with minimum residual
[~, label] = min(residuals); 



