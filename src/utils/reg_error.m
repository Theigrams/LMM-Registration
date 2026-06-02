function [eR, eT] = reg_error(GrtR, GrtT, Ts)
% REG_ERROR  Mean rotation and translation error of a multi-view registration.
%
%   [eR, eT] = REG_ERROR(GrtR, GrtT, Ts) compares the estimated transforms
%   against the ground truth. Both are expressed relative to the first view,
%   so the metric is invariant to the choice of global frame.
%
%     eR = (1/M) sum_i arccos( (tr(R1_i' * R2_i) - 1) / 2 )   [radians]
%     eT = (1/M) sum_i || t1_i - t2_i ||_2
%
%   GrtR - 1xM cell of ground-truth 3x3 rotations
%   GrtT - 1xM cell of ground-truth 3x1 translations
%   Ts   - 4x4xM estimated transformations

M = size(Ts, 3);
eR = 0;
eT = 0;

GR = GrtR{1};  Gt = GrtT{1};        % ground-truth reference (view 1)
R0 = Ts(1:3, 1:3, 1);  t0 = Ts(1:3, 4, 1);   % estimated reference (view 1)

for i = 1:M
    R1 = GR' * GrtR{i};             % ground truth, relative to view 1
    t1 = GR' * (GrtT{i} - Gt);
    R2 = R0' * Ts(1:3, 1:3, i);     % estimate, relative to view 1
    t2 = R0' * (Ts(1:3, 4, i) - t0);
    eR = eR + real(acos(0.5 * (trace(R1' * R2) - 1)));
    eT = eT + norm(t1 - t2, 2);
end

eR = eR / M;
eT = eT / M;
end
