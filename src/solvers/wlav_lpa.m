function [R, t] = wlav_lpa(P, Q, w)
% WLAV_LPA  Weighted least-absolute-value alignment via Linear Programming Approximation.
%
%   [R, t] = WLAV_LPA(P, Q, w) solves the WLAV sub-problem of the M-step
%
%       min_{R,t}  sum_i w_i * || R*p_i + t - q_i ||_1
%       s.t.       R in SO(3)
%
%   using the LPA scheme of the paper (Sec. "LPA Method"). The rotation is
%   linearised by the first-order exponential map R ~= I + [r]_x, which
%   turns the objective into ||A*x - b||_1 with x = [r; t]. This L1 problem
%   is solved by an interior point method, and the resulting increment is
%   re-projected onto SO(3) by a final SVD step.
%
%   P - 3xN matrix, source points
%   Q - 3xN matrix, target points
%   w - Nx1 vector of non-negative weights
%
%   See also WLAV_ADMM, WLS_SVD, L1_INTERIOR_POINT.

N = size(P, 2);

% Warm start from the weighted least-squares (L2) solution.
[R, t] = wls_svd(P, Q, w);
P = R * P + t;

% Build the linearised system A*x - b, with x = [r; t] (Eq. for linear_deviation).
% Each correspondence contributes 3 rows:  w_i*( -[p_i]_x | I ) * [r; t] = -w_i*(p_i - q_i)
A = zeros(3 * N, 6);
b = zeros(3 * N, 1);
for i = 1:N
    rows = 3 * i - 2 : 3 * i;
    A(rows, 1:3) = -w(i) * skew(P(:, i));
    A(rows, 4:6) =  w(i) * eye(3);
    b(rows)      = -w(i) * (P(:, i) - Q(:, i));
end

% Solve  min_x ||A*x - b||_1  by the interior point method.
x0 = ones(6, 1) * 1e-5;
gfun  = @(z) A * z;
gtfun = @(z) A' * z;
xp = l1_interior_point(x0, gfun, gtfun, b, 1e-3, 25, 1e-8, 200);

% Recover the rigid increment and re-project onto SO(3).
r  = xp(1:3);
dR = eye(3) + skew(r);     % first-order exponential map
dt = xp(4:6);
[dR, dt] = wls_svd(P, dR * P + dt, w);

R = dR * R;
t = dR * t + dt;
end
