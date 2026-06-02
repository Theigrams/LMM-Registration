function [R, t] = wlav_admm(P, Q, w)
% WLAV_ADMM  Weighted least-absolute-value alignment via ADMM.
%
%   [R, t] = WLAV_ADMM(P, Q, w) solves the WLAV sub-problem of the M-step
%
%       min_{R,t}  sum_i w_i * || R*p_i + t - q_i ||_1
%       s.t.       R in SO(3)
%
%   using the ADMM scheme of the paper (Sec. "ADMM Method"). Introducing the
%   split variable z_i = w_i*(R*p_i + t - q_i), the iterations alternate
%   between (i) a soft-thresholding (shrinkage) update of z, and (ii) a
%   weighted least-squares update of (R, t) solved in closed form by SVD.
%
%   P - 3xN matrix, source points
%   Q - 3xN matrix, target points
%   w - Nx1 vector of non-negative weights
%
%   See also WLAV_LPA, WLS_SVD.

max_iter = 50;
rho = 10;            % augmented Lagrangian penalty
alpha = 1.2;         % penalty growth factor
max_rho = 1e5;
min_rho = 1e-1;
stop_threshold = 1e-2;

w = w(:)';
keep = (w > 1e-10);  % drop negligible-weight correspondences
P = P(:, keep);
Q = Q(:, keep);
w = w(keep);

Lambda = zeros(size(P));        % scaled dual variable
[R, t] = wls_svd(P, Q, w');     % warm start from L2 solution

for k = 1:max_iter
    S = w .* (R * P - Q + t);                 % s_i = w_i*(R*p_i + t - q_i)

    % z-update: soft-threshold (shrinkage operator S_{1/rho}).
    Z = shrink(S - Lambda / rho, 1 / rho);

    % (R, t)-update: weighted least squares against the relaxed targets.
    Qhat = Q + (Z + Lambda / rho) ./ w;
    [R, t] = wls_svd(P, Qhat, (w.^2)');

    % Dual update.
    Lambda = Lambda + rho * (Z - S);

    if rho < max_rho && rho > min_rho
        rho = rho * alpha;
    end
    if norm(Z - S) < stop_threshold
        break
    end
end
end

function Y = shrink(X, kappa)
% Element-wise soft-thresholding (shrinkage) operator S_kappa(x).
Y = zeros(size(X));
Y(X > kappa)  = X(X > kappa)  - kappa;
Y(X < -kappa) = X(X < -kappa) + kappa;
end
