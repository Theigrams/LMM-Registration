function [x, res, iter] = cg_solve(A, b, tol, maxiter, verbose)
% CG_SOLVE  Solve a symmetric positive-definite system A*x = b by conjugate gradients.
%
%   [x, res, iter] = CG_SOLVE(A, b, tol, maxiter, verbose)
%
%   A       - NxN matrix or a function handle implementing the product A*x
%   b       - N vector
%   tol     - desired precision; stops when norm(A*x-b)/norm(b) < tol
%   maxiter - maximum number of iterations
%   verbose - 0 to stay silent, otherwise print progress every 'verbose' iters
%
%   Adapted from l1-magic by Justin Romberg, Caltech (2005).

if nargin < 5, verbose = 0; end

implicit = isa(A, 'function_handle');

x = zeros(length(b), 1);
r = b;
d = r;
delta = r' * r;
delta0 = b' * b;
numiter = 0;
bestx = x;
bestres = sqrt(delta / delta0);

while (numiter < maxiter) && (delta > tol^2 * delta0)
    if implicit, q = A(d); else, q = A * d; end
    alpha = delta / (d' * q);
    x = x + alpha * d;

    if mod(numiter + 1, 50) == 0
        if implicit, r = b - A(x); else, r = b - A * x; end
    else
        r = r - alpha * q;
    end

    deltaold = delta;
    delta = r' * r;
    beta = delta / deltaold;
    d = r + beta * d;
    numiter = numiter + 1;

    if sqrt(delta / delta0) < bestres
        bestx = x;
        bestres = sqrt(delta / delta0);
    end

    if verbose && mod(numiter, verbose) == 0
        fprintf('cg: iter = %d, best res = %8.3e, cur res = %8.3e\n', ...
            numiter, bestres, sqrt(delta / delta0));
    end
end

x = bestx;
res = bestres;
iter = numiter;
end
