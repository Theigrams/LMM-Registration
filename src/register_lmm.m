function [Ts, info] = register_lmm(views, Ts, ns, opts)
% REGISTER_LMM  Robust multi-view point set registration with a Laplacian Mixture Model.
%
%   [Ts, info] = REGISTER_LMM(views, Ts, ns, opts)
%
%   Simultaneously aligns M point sets by assuming each transformed point is
%   generated from a Laplacian Mixture Model (LMM) whose centres are the
%   corresponding points in the other views. The model parameters and the
%   rigid transformations are estimated with an EM scheme: the E-step builds
%   point correspondences (kd-tree nearest neighbour under the L1 / cityblock
%   metric) and the posterior responsibilities, and the M-step solves a
%   weighted least-absolute-value (WLAV) problem per view. Because the
%   Laplacian uses the sparsity-inducing L1 norm, the method is robust to
%   heavy-tailed noise and outliers without an explicit uniform component.
%
%   Inputs
%     views - 1xM cell array, views{i} is a 3xNi matrix of points already
%             expressed in the (initial) common reference frame.
%     Ts    - 4x4xM array of initial transformations (views{i} = T_i applied
%             to the raw i-th point set).
%     ns    - 1xM cell array of kd-tree searchers (createns) built on the
%             raw i-th point set with the cityblock distance.
%     opts  - struct with optional fields:
%               .solver   'lpa' (default) or 'admm'   WLAV sub-problem solver
%               .max_iter maximum EM iterations        (default 100)
%               .tol      relative stopping tolerance  (default 1e-5)
%               .verbose  print per-iteration progress (default true)
%
%   Outputs
%     Ts   - 4x4xM refined transformations, re-anchored so that the first
%            view sits at the origin.
%     info - struct with fields .iters, .obj, .b (final objective and scale).
%
%   Reference
%     J. Zhang, M. Zhao, X. Jiang, D.-M. Yan, "Robust Multi-view Registration
%     of Point Sets with Laplacian Mixture Model".
%
%   See also WLAV_LPA, WLAV_ADMM.

if nargin < 4, opts = struct(); end
if ~isfield(opts, 'solver'),  opts.solver  = 'lpa'; end
if ~isfield(opts, 'max_iter'), opts.max_iter = 100; end
if ~isfield(opts, 'tol'),     opts.tol     = 1e-5; end
if ~isfield(opts, 'verbose'), opts.verbose = true; end

switch lower(opts.solver)
    case 'lpa',  solve_wlav = @wlav_lpa;
    case 'admm', solve_wlav = @wlav_admm;
    otherwise,   error('register_lmm:solver', 'Unknown solver "%s".', opts.solver);
end

M = size(Ts, 3);
N = cellfun(@(v) size(v, 2), views);   % cardinality of each view
b = ones(1, M);                        % per-view Laplacian scale parameter
obj_hist = [];
rel_change = inf;
iter = 0;

while rel_change > opts.tol && iter < opts.max_iter
    iter = iter + 1;
    L = zeros(1, M);

    for i = 1:M
        Pi = views{i};
        Q = zeros(3, N(i), M - 1);     % corresponding points from the other views
        D = zeros(N(i), M - 1);        % L1 distances to those correspondences

        % ---- E-step: correspondences via nearest neighbour in each other view ----
        for j = 1:M
            if j == i, continue; end
            Pi_in_j = pc_transform(Pi, inv_transform(Ts(:, :, j)));   % i-th view in j's frame
            [idx, d_ij] = knnsearch(ns{j}, Pi_in_j');
            col = j - (j > i);                                        % skip the i==j slot
            Q(:, :, col) = views{j}(:, idx);
            D(:, col) = d_ij;
        end

        % Posterior responsibilities alpha (Laplacian density beta, normalised).
        b_other = b([1:i-1, i+1:M]);
        beta = exp(-D ./ b_other) ./ (2 * b_other).^3;
        alpha = beta ./ (sum(beta, 2) + 5e-4);

        % ---- M-step: weighted L1 alignment of view i against its correspondences ----
        P = repmat(Pi, 1, M - 1);
        Qs = reshape(Q, 3, N(i) * (M - 1));
        w = reshape(alpha, N(i) * (M - 1), 1);
        [dR, dt] = solve_wlav(P, Qs, w);

        dT = [dR, dt; zeros(1, 3), 1];
        Ts(:, :, i) = dT * Ts(:, :, i);
        views{i} = pc_transform(views{i}, dT);

        % Weighted mean residual -> update of the per-view scale b.
        L(i) = sum(sum(abs(dR * P - Qs + dt) .* w')) / (3 * N(i));
        b(i) = max(L(i), 1e-8);
    end

    % Negative log-likelihood surrogate (scaled for readability).
    obj = 3 * (log(2 * L) + 1) * N' / 100;
    obj_hist(end + 1) = obj; %#ok<AGROW>
    if iter > 1
        rel_change = abs(obj_hist(iter - 1) - obj_hist(iter)) / abs(obj_hist(iter - 1));
    end
    if opts.verbose
        fprintf('iter %3d | obj = %f | b = %f\n', iter, obj, mean(b));
    end
end

% Re-anchor the solution so that the first view is the reference frame.
R0 = inv_transform(Ts(:, :, 1));
for i = 1:M
    Ts(:, :, i) = R0 * Ts(:, :, i);
end

info = struct('iters', iter, 'obj', obj_hist(end), 'b', mean(b));
end
