function Q = pc_transform(P, T)
% PC_TRANSFORM  Apply a rigid transformation to a point cloud.
%
%   Q = PC_TRANSFORM(P, T) transforms the 3xN point cloud P by the 4x4
%   homogeneous transformation T, returning the 3xN result Q = R*P + t.
%
%   P - 3xN matrix, source points (one point per column)
%   T - 4x4 homogeneous transformation matrix
%
%   See also INV_TRANSFORM.

R = T(1:3, 1:3);
t = T(1:3, 4);
Q = R * P + t;
end
