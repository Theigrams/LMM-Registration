% demo.m
% ------------------------------------------------------------------------
% Robust multi-view registration with a Laplacian Mixture Model (LMM).
%
% This demo aligns the 10 partial views of the Stanford Bunny shipped in
% data/bunny.mat. Each view is contaminated with noise and outliers and
% given a perturbed initial pose; REGISTER_LMM recovers the rigid
% transformations that bring all views into a common frame.
% ------------------------------------------------------------------------
clc; clear; close all;
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), 'src')));

%% --- Settings ---------------------------------------------------------
solver   = 'admm';     % 'admm' (faster, recommended) or 'lpa' (most accurate)
max_iter = 100;        % EM iterations
init_pert = 0.025;     % magnitude of the random initial-pose perturbation (rad)
rng(0);                % reproducible perturbation

%% --- Load data --------------------------------------------------------
% bunny.mat provides:
%   model : 1xM cell, model{i} is an Ni x 3 partial view (set-centered frame)
%   shape : 1xM cell, dense per-view shape used only for visualization
%   GrtR  : 1xM cell of ground-truth 3x3 rotations
%   GrtT  : 1xM cell of ground-truth 3x1 translations
%   N     : number of views
data = load(fullfile(fileparts(mfilename('fullpath')), 'data', 'bunny.mat'));
model = data.model;  shape = data.shape;
GrtR = data.GrtR;    GrtT = data.GrtT;
M = data.N;

% The raw views live at ~1000x the dense shapes; registration is run in the
% shape's unit scale so the Laplacian scale parameter b stays well-conditioned.
norm_scale = 1000;
vis_slab   = [1, -0.02, -0.019];   % [axis, vmin, vmax] for the cross-section plot

%% --- Build perturbed initial poses and kd-trees -----------------------
Ts = zeros(4, 4, M);
views = cell(1, M);
ns = cell(1, M);
for i = 1:M
    pts = model{i} / norm_scale;       % normalize to unit scale (Ni x 3)

    R_pert = random_rotation(init_pert);
    Ts(1:3, 1:3, i) = R_pert * GrtR{i};
    Ts(1:3, 4, i)   = GrtT{i};
    Ts(4, 4, i)     = 1;

    % kd-tree on the raw view (L1 / cityblock metric, as used in the E-step).
    ns{i} = createns(pts, 'NSMethod', 'kdtree', 'Distance', 'cityblock');
    % view points expressed in the common reference frame.
    views{i} = pc_transform(pts', Ts(:, :, i));
end

%% --- Register ---------------------------------------------------------
opts = struct('solver', solver, 'max_iter', max_iter);
tic;
[Ts_est, info] = register_lmm(views, Ts, ns, opts);
elapsed = toc;

%% --- Report -----------------------------------------------------------
[eR0, eT0] = reg_error(GrtR, GrtT, Ts);
[eR,  eT]  = reg_error(GrtR, GrtT, Ts_est);
fprintf('\nSolver: %s   |   iterations: %d   |   time: %.2f s\n', solver, info.iters, elapsed);
fprintf('Initial   error:  R = %.6f rad,  t = %.6f\n', eR0, eT0);
fprintf('Registered error: R = %.6f rad,  t = %.6f\n', eR, eT);

%% --- Visualize --------------------------------------------------------
figure('Name', 'Aligned model');
Model = build_aligned_model(shape, Ts_est, 1);   % dense shapes are already unit-scale
title(sprintf('LMM-%s aligned model', upper(solver)));

cross_section(Model, vis_slab(1), vis_slab(2), vis_slab(3));
title('Cross section of the aligned model');

% ----------------------------------------------------------------------
function R = random_rotation(r)
% Small random rotation: independent angles in [-r, r] about x, y, z.
a = r * (2 * rand - 1);  b = r * (2 * rand - 1);  c = r * (2 * rand - 1);
Rx = [1 0 0; 0 cos(a) -sin(a); 0 sin(a) cos(a)];
Ry = [cos(b) 0 sin(b); 0 1 0; -sin(b) 0 cos(b)];
Rz = [cos(c) -sin(c) 0; sin(c) cos(c) 0; 0 0 1];
R = Rz * Ry * Rx;
end
