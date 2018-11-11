function [label, iterationCount] = SRC(tr, te, epsilon, lambda, options)
% This fuction implement SRC algorithm
% Input: 
%       tr.x        training samples of size m*n
%       tr.y        class label of training 
%       te.x        a test sample of size m*k    
%       epsilon     optimal error tolerance epsilon
%       lambda      coefficient controls data fidelity
% Output:
%       x           sparse representation, x \in R^{n*1}
% Reference
%       [1] J. Wright, et al. Robust Face Recognition via Sparse Representation
%       [2] https://github.com/hiroyuki-kasai/ClassifierToolbox
%       [3] L. Zhang, et all. A Simple Homotopy Algorithm for Compressive Sensing

% get the size of datesets
[~, n] = size(tr.x);
[~, k] = size(te.x);

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
    te.x  =  disc_set' * te.x;
end

% normalize the columns of A to have unit l2-norm
tr.x = tr.x ./ sqrt(sum(tr.x .^ 2, 2));
te.x = te.x ./ sqrt(sum(te.x .^ 2, 2));

% get the class label
classes = unique(tr.y);
class_num = length(classes);

% solve the l1-minimization problem
P = tr.x'/(tr.x'*tr.x+1e-5*eye(n));
label = zeros(1,k);
iterationCount = zeors(1,k);
for i = 1:k
    % tex is a sample to predict
    tex = te.x(:,i);
    x0 = P*tex;

    % implement Homotopy algorithm
    [xp, iterationCount(i)] = SolveHomotopy(tr.x, tex, ...
                            'maxIteration', 5000,...
                            'isNonnegative', false, ...
                            'stoppingCriterion', -1, ...
                            'groundtruth', x0, ...
                            'lambda', lambda, ...
                            'tolerance', epsilon);  

    % calculate residual for each class
    residuals = zeros(1, max(classes));
    for j = 1 :class_num
        idx = find(tr.y == classes(j));
        residuals(round(classes(j))) = norm(te.x - tr.x(:,idx)*xp(idx));
    end

    % get the predicted label with minimum residual
    [~, label(i)] = min(residuals); 
end


