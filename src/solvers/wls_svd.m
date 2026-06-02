function [R, t] = wls_svd(P, Q, w)
% WLS_SVD  Weighted least-squares rigid alignment with SO(3) constraint.
%
%   [R, t] = WLS_SVD(P, Q, w) solves the weighted Procrustes problem
%
%       min_{R,t}  sum_i w_i * || R*p_i + t - q_i ||_2^2
%       s.t.       R in SO(3)
%
%   in closed form via SVD. It is used both as a sub-step of the ADMM
%   solver and to provide the initial guess for the LPA solver.
%
%   P - 3xN matrix, source points
%   Q - 3xN matrix, target points
%   w - Nx1 (or 1xN) vector of non-negative weights
%
%   See also WLAV_LPA, WLAV_ADMM.

w = w(:)';                       % row vector
keep = (w > 1e-10);              % drop negligible-weight correspondences
P = P(:, keep);
Q = Q(:, keep);
w = w(keep);

sum_w = sum(w);
p_mean = sum(w .* P, 2) / sum_w;
q_mean = sum(w .* Q, 2) / sum_w;
X = P - p_mean;
Y = Q - q_mean;

% Weighted cross-covariance and its SVD.
S = (X .* w) * Y';
[U, ~, V] = svd(S);
R = V * U';
if det(R) < 0                    % reflection guard: enforce a proper rotation
    D = eye(3);
    D(3, 3) = -1;
    R = V * D * U';
end
t = q_mean - R * p_mean;
end
