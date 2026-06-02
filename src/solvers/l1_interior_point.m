function xp = l1_interior_point(x0, A, At, y, pdtol, pdmaxiter, cgtol, cgmaxiter)
% L1_INTERIOR_POINT  Minimise ||A*x - y||_1 by a primal-dual interior point method.
%
%   xp = L1_INTERIOR_POINT(x0, A, At, y, pdtol, pdmaxiter, cgtol, cgmaxiter)
%
%   Solves   min_x ||A*x - y||_1, recast as the linear program
%
%       min_{x,u} sum(u)   s.t.   A*x - u - y <= 0
%                                -A*x - u + y <= 0
%
%   and solved with a primal-dual interior point method. When A is given as
%   a function handle the Newton systems are solved with conjugate gradients
%   (see CG_SOLVE); when A is a matrix they are solved directly.
%
%   x0        - Nx1 initial point
%   A         - MxN matrix, or a handle to a function returning A*x
%   At        - handle returning A'*v (ignored when A is a matrix)
%   y         - Mx1 observations (M > N)
%   pdtol     - duality-gap tolerance              (default 1e-3)
%   pdmaxiter - max primal-dual iterations          (default 50)
%   cgtol     - conjugate-gradient tolerance        (default 1e-8)
%   cgmaxiter - max conjugate-gradient iterations   (default 200)
%
%   Adapted from l1-magic by Justin Romberg, Caltech (2005).

largescale = isa(A, 'function_handle');

if nargin < 5, pdtol = 1e-3; end
if nargin < 6, pdmaxiter = 50; end
if nargin < 7, cgtol = 1e-8; end
if nargin < 8, cgmaxiter = 200; end

N = length(x0);
M = length(y);

alpha = 0.01;
beta = 0.5;
mu = 10;

gradf0 = [zeros(N, 1); ones(M, 1)];

x = x0;
if largescale, Ax = A(x); else, Ax = A * x; end
u = 0.95 * abs(y - Ax) + 0.10 * max(abs(y - Ax));

fu1 = Ax - y - u;
fu2 = -Ax + y - u;

lamu1 = -1 ./ fu1;
lamu2 = -1 ./ fu2;

if largescale, Atv = At(lamu1 - lamu2); else, Atv = A' * (lamu1 - lamu2); end

sdg = -(fu1' * lamu1 + fu2' * lamu2);   % surrogate duality gap
tau = mu * 2 * M / sdg;

rcent = [-lamu1 .* fu1; -lamu2 .* fu2] - (1 / tau);
rdual = gradf0 + [Atv; -lamu1 - lamu2];
resnorm = norm([rdual; rcent]);

pditer = 0;
done = (sdg < pdtol) || (pditer >= pdmaxiter);
while ~done
    pditer = pditer + 1;

    w2 = -1 - 1 / tau * (1 ./ fu1 + 1 ./ fu2);

    sig1 = -lamu1 ./ fu1 - lamu2 ./ fu2;
    sig2 = lamu1 ./ fu1 - lamu2 ./ fu2;
    sigx = sig1 - sig2.^2 ./ sig1;

    if largescale
        w1 = -1 / tau * At(-1 ./ fu1 + 1 ./ fu2);
        w1p = w1 - At((sig2 ./ sig1) .* w2);
        h11pfun = @(z) At(sigx .* A(z));
        [dx, cgres] = cg_solve(h11pfun, w1p, cgtol, cgmaxiter, 0);
        if cgres > 1/2
            disp('Cannot solve system. Returning previous iterate.');
            xp = x; return
        end
        Adx = A(dx);
    else
        w1 = -1 / tau * (A' * (-1 ./ fu1 + 1 ./ fu2));
        w1p = w1 - A' * ((sig2 ./ sig1) .* w2);
        H11p = A' * (sparse(diag(sigx)) * A);
        opts.POSDEF = true; opts.SYM = true;
        [dx, hcond] = linsolve(H11p, w1p, opts);
        if hcond < 1e-14
            disp('Matrix ill-conditioned. Returning previous iterate.');
            xp = x; return
        end
        Adx = A * dx;
    end

    du = (w2 - sig2 .* Adx) ./ sig1;

    dlamu1 = -(lamu1 ./ fu1) .* (Adx - du) - lamu1 - (1 / tau) * 1 ./ fu1;
    dlamu2 = (lamu2 ./ fu2) .* (Adx + du) - lamu2 - (1 / tau) * 1 ./ fu2;
    if largescale, Atdv = At(dlamu1 - dlamu2); else, Atdv = A' * (dlamu1 - dlamu2); end

    % Keep the step feasible: lamu1, lamu2 > 0 and fu1, fu2 < 0.
    indl = dlamu1 < 0; indu = dlamu2 < 0;
    s = min([1; -lamu1(indl) ./ dlamu1(indl); -lamu2(indu) ./ dlamu2(indu)]);
    indl = (Adx - du) > 0; indu = (-Adx - du) > 0;
    s = 0.99 * min([s; -fu1(indl) ./ (Adx(indl) - du(indl)); ...
                       -fu2(indu) ./ (-Adx(indu) - du(indu))]);

    % Backtracking line search.
    suffdec = 0;
    backiter = 0;
    while ~suffdec
        xp = x + s * dx;  up = u + s * du;
        Axp = Ax + s * Adx;  Atvp = Atv + s * Atdv;
        lamu1p = lamu1 + s * dlamu1;  lamu2p = lamu2 + s * dlamu2;
        fu1p = Axp - y - up;  fu2p = -Axp + y - up;
        rdp = gradf0 + [Atvp; -lamu1p - lamu2p];
        rcp = [-lamu1p .* fu1p; -lamu2p .* fu2p] - (1 / tau);
        suffdec = norm([rdp; rcp]) <= (1 - alpha * s) * resnorm;
        s = beta * s;
        backiter = backiter + 1;
        if backiter > 32
            disp('Stuck backtracking. Returning last iterate.');
            xp = x; return
        end
    end

    % Commit the step.
    x = xp;  u = up;
    Ax = Axp;  Atv = Atvp;
    lamu1 = lamu1p;  lamu2 = lamu2p;
    fu1 = fu1p;  fu2 = fu2p;

    sdg = -(fu1' * lamu1 + fu2' * lamu2);
    tau = mu * 2 * M / sdg;
    rcent = [-lamu1 .* fu1; -lamu2 .* fu2] - (1 / tau);
    rdual = rdp;
    resnorm = norm([rdual; rcent]);

    done = (sdg < pdtol) || (pditer >= pdmaxiter);
end
end
