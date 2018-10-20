clear all
close all
clc

M			= 80;                                           % --- Number of discretization points along x
N			= 100;                                          % --- Number of discretization points along y
numIter		= 2000;                                         % --- Number of iterations
dx			= 1. / (M - 1.);                                % --- Discretization step along x
dy			= 1. / (N - 1.);                                % --- Discretization step along y

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DISCRETIZATION POINTS ALONG X %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x           = (0 : (M - 1)) * dx;
y           = (0 : (N - 1)) * dy;

[X, Y]      = meshgrid(x, y);

%%%%%%%%%%%%%%%
% SOURCE TERM %
%%%%%%%%%%%%%%%
F           = X.^2 + Y.^2;	

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZING THE SOLUTION WITH BOUNDARY CONDITIONS %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
U           = zeros(size(X));
U_OLD       = zeros(size(X));

% --- Upper boundary condition
U(N, :)     = x.^2 / 2;
U_OLD(N, :) = x.^2 / 2;
% --- Lower boundary condition
U(1, :)     = 0;
U_OLD(1, :) = 0;
% --- Left boundary condition
U(:, 1)     = sin(pi * y);
U_OLD(:, 1) = sin(pi * y);
% --- Right boundary condition
U(:, M)     = exp(pi) * sin(pi * y) + y.^2 / 2;
U_OLD(:, M) = exp(pi) * sin(pi * y) + y.^2 / 2;

%%%%%%%%%%%%%%%%%%%%%%
% REFERENCE SOLUTION %
%%%%%%%%%%%%%%%%%%%%%%
U_REF = exp(pi * X) .* sin(pi * Y) + 0.5 * (X .* Y).^2;

%%%%%%%%%%%%%%
% ITERATIONS %
%%%%%%%%%%%%%%
for k = 1 : numIter
    
    U_OLD = jacobiIterator(U, U_OLD, F, dx * dy, M, N);

    % --- Pointers swap
    TEMP    = U;
    U       = U_OLD;
    U_OLD   = TEMP;
    
%     figure(1)
%     h = surf(x, y, U, 'EdgeColor', 'none');       
%     shading interp
%     axis([0 1 0 1 min(min(U_REF)) max(max(U_REF))])
%     title('2-D Poisson equation')
%     xlabel('x')
%     ylabel('y')
%     zlabel('Numerical solution')
%     fprintf("Iteration number %i\n", k);
%     drawnow

end

% --- Loading the CUDA result
load d_result.txt
% --- Transposition is necessary since CUDA writes the matrix row-wise,
% while Matlab reads it column-wise.
d_result = reshape(d_result, M, N).';

function U_OLD = jacobiIterator(U, U_OLD, F, delta2, M, N)

U_OLD(2 : N - 1, 2 : M - 1) = (U(2 : N - 1, 1 : M - 2) + ...
                               U(2 : N - 1, 3 : M    ) + ...
                               U(1 : N - 2, 2 : M - 1) + ...
                               U(3 : N    , 2 : M - 1) + ...
                               delta2 * F(2 : N - 1, 2 : M - 1)) * 0.25;

end
