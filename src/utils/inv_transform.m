function invT = inv_transform(T)
% INV_TRANSFORM  Inverse of a 4x4 rigid transformation.
%
%   invT = INV_TRANSFORM(T) returns the inverse of the homogeneous rigid
%   transformation T = [R t; 0 1], computed in closed form as
%   [R' -R'*t; 0 1] to avoid a generic matrix inverse.
%
%   See also PC_TRANSFORM.

R = T(1:3, 1:3);
t = T(1:3, 4);
invT = [R', -R' * t; zeros(1, 3), 1];
end
