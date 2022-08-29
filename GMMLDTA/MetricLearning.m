function A = MetricLearning(X, y, t, params)
% Geometric mean metric learning algorithm
%X��ÿһ�д���һ�������������ڼ���ʱX��ÿһ�д���һ������
if (~exist('params')),
    params = struct();
end
params = SetDefaultParams(params);
if (~exist('t')),
    t = 0.5;
end
   if istable(X)==1
      X = table2array(X);
   end 

%the number of constraints
hk = length(unique(y));
num_const = params.const_factor * (hk * (hk-1));

%constraint generation
[S, D, U] = ConstGen(X, y, num_const);
lamda = 1;
S = S + lamda .* U;
D = D + lamda .* U;

if isfield(params, 'A0') ~= 0
    %regularization for the cases that we have a prior knowledge
    S = S + params.mu / params.A0;
    D = D + params.mu * params.A0;
elseif (rcond(S) < params.thresh) || (rcond(D) < params.thresh)
    %auto-regularization for the cases where S or D are near-singular
    S = S + params.mu * eye(length(S));
    D = D + params.mu * eye(length(S));
end
S = S + params.mu * eye(length(S));
D = D + params.mu * eye(length(S));
%computing the matrix
A = real(sharp(inv(S),D,t)); %computing the geodesic using the Cholesky-Schur method
% A = real(S\((S*D)^t));
end
