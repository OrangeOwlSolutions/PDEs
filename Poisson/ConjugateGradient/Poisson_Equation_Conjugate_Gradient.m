clear all
close all
clc

M               = 16;                                   % --- Number of discretization points along x
N               = 16;                                   % --- Number of discretization points along y
xM              = 1;                                    % --- Semi-extent, along x, of the analysis region
yM              = 1;                                    % --- Semi-extent, along y, of the analysis region
dx              = 2 * xM / (M + 1);                     % --- Discretization step along x
dy              = 2 * yM / (N + 1);                     % --- Discretization step along y

%%%%%%%%%%%%%%%%%%%%%%%
% DISCRETIZATION GRID %
%%%%%%%%%%%%%%%%%%%%%%%
x               = -(xM - dx) : dx : (xM - dx);
y               = -(yM - dy) : dy : (yM - dy);
[X, Y]          = meshgrid(x, y);

%%%%%%%%%%%%%%%
% SOURCE TERM %
%%%%%%%%%%%%%%%
% --- In this case, the solution is U = sin(pi * X) .* sin(pi * Y)
F               = reshape(-2 * pi^2 * sin(pi * X) .* sin(pi * Y), M * N, 1);
% F               = ones(M * N, 1);

%%%%%%%%%%%%%%%%%%%%%%%
% MATRIX CONSTRUCTION %
%%%%%%%%%%%%%%%%%%%%%%%
e               = ones(M * N, 1);
e1              = e;
e2              = e;
e1(N     : N : end) = 0;
e2(N + 1 : N : end) = 0;
cmn             = -2 / dx^2 - 2 / dy^2;
A               = spdiags([e / dy^2 e1 / dx^2 cmn * e e2 / dx^2 e / dy^2], [-N -1 0 1 N], M * N, M * N);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CONJUGATE GRADIENT RECONSTRUCTION %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
U               = reshape(conjugateGradient(A, F, 1e-10), N, M);

%%%%%%%%%%
% GRAPHS %
%%%%%%%%%%
figure(1)
h = surf(x, y, U, 'EdgeColor', 'none');       
shading interp
axis([-1 1 -1 1 min(min(U)) max(max(U))])
title('2-D Poisson equation')
xlabel('x')
ylabel('y')
zlabel('Numerical solution')
