function [label, iterationCount] = SRC(tr, te, epsilon, lambda, options)
% This fuction implement SRC algorithm
% Input: 
%       tr.X            training samples of size m*n
%       tr.y            class label of training 
%       te.X            a test sample of size m*k    
%       epsilon         optimal error tolerance epsilon
%       lambda          coefficient controls data fidelity
% Output:
%       label           sparse representation of size R^{n*1}
%       iterationCount  the number of iterations until termination
% Reference
%       [1] J. Wright, et al. Robust Face Recognition via Sparse Representation
%       [2] https://github.com/hiroyuki-kasai/ClassifierToolbox
%       [3] L. Zhang, et all. A Simple Homotopy Algorithm for Compressive Sensing

    % get the size of datesets
    [~, n] = size(tr.X);

    % extract options    
    if ~isfield(options, 'eigenface')
        eigenface = false;
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
        [disc_set, ~, ~] = Eigenface_f(tr.X, eigenface_dim);
        tr.X  =  disc_set' * tr.X;
        te.X  =  disc_set' * te.X;
    end
    
    [~, n] = size(tr.X);
    [~, k] = size(te.X);
    % normalize the columns of A to have unit l2-norm
    tr.X = tr.X./repmat(sqrt(sum(tr.X.^2)),[size(tr.X,1),1]);
    te.X = te.X./repmat(sqrt(sum(te.X.^2)),[size(te.X,1),1]);
    
    % get the class label
    classes = unique(tr.y);
    class_num = length(classes);

    % solve the l1-minimization problem
    P = (tr.X'\(tr.X'*tr.X+1e-5*eye(n)))';
    label = zeros(1,k);
    iterationCount = zeros(1,k);
    for i = 1:k
        % tex is a sample to predict
        tex = te.X(:,i);
        x0 = P*tex;

        % implement Homotopy algorithm
        [xp, iterationCount(i)] = SolveHomotopy(tr.X, tex, ...
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
            residuals(round(classes(j))) = norm(tex - tr.X(:,idx)*xp(idx));
        end

        % get the predicted label with minimum residual
        [~, label(i)] = min(residuals); 
    end
end


