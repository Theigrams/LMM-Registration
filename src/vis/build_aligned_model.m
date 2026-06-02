function Model = build_aligned_model(shapes, Ts, scale)
% BUILD_ALIGNED_MODEL  Assemble and plot the aligned multi-view model.
%
%   Model = BUILD_ALIGNED_MODEL(shapes, Ts, scale) transforms every per-view
%   dense shape by its estimated transformation, plots each view in a
%   distinct colour, and returns the concatenated 3xK point cloud.
%
%   shapes - 1xM cell, shapes{i} is an Ni x 3 dense point set for view i
%   Ts     - 4x4xM estimated transformations
%   scale  - scalar applied to the shapes before transforming (default 1)
%
%   See also CROSS_SECTION.

if nargin < 3, scale = 1; end

M = size(Ts, 3);
colours = lines(max(M, 7));   % distinct colour per view

Model = [];
hold on
for i = 1:M
    pts = pc_transform(shapes{i}' * scale, Ts(:, :, i));
    Model = [Model, pts]; %#ok<AGROW>
    plot3(pts(1, :), pts(2, :), pts(3, :), '.', 'Color', colours(i, :), 'MarkerSize', 1);
end
axis equal
axis off
view(3)
end
