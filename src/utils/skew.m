function S = skew(r)
% SKEW  Skew-symmetric matrix of a 3-vector.
%
%   S = SKEW(r) returns the 3x3 skew-symmetric matrix [r]_x such that
%   [r]_x * v = cross(r, v) for any 3-vector v. This is the matrix used in
%   the exponential map R = exp([r]_x) of SO(3).
%
%   r - 3x1 vector

S = [    0,  -r(3),   r(2);
      r(3),     0,  -r(1);
     -r(2),  r(1),     0  ];
end
