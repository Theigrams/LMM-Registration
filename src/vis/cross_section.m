function cross_section(Model, axisIdx, vmin, vmax)
% CROSS_SECTION  Plot a thin slab of the aligned model as a 2-D cross section.
%
%   CROSS_SECTION(Model, axisIdx, vmin, vmax) keeps the points whose
%   coordinate along axis 'axisIdx' lies in [vmin, vmax] and scatters them in
%   the plane of the remaining two axes. A well-registered model shows the
%   per-view slices collapsing onto a single thin contour.
%
%   Model   - 3xK aligned point cloud (see BUILD_ALIGNED_MODEL)
%   axisIdx - slicing axis (1 = x, 2 = y, 3 = z)
%   vmin    - lower bound of the slab along axisIdx
%   vmax    - upper bound of the slab along axisIdx

other = [1, 2, 3];
other(axisIdx) = [];

slab = Model(:, Model(axisIdx, :) < vmax & Model(axisIdx, :) > vmin);

figure
plot(slab(other(2), :), slab(other(1), :), '.b', 'MarkerSize', 2);
axis equal
axis off
end
