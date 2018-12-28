clear all
close all
clc

% Solve the Poisson equation u_{xx} + u_{yy} = f(x,y) on [a, b] x [a, b].  

f = @(x,y) 1.25 * exp(x + y / 2);       % --- rhs function

a = 0;                                  % --- Left boundary of the solution domain 
b = 1;                                  % --- Right boundary of the solution domain
N_internal = 20;                        % --- Number of internal discretization points along each dimension
h = (b - a) / (N_internal + 1);         % --- Sampling interval
x = linspace(a, b, N_internal + 2);     % --- Sampling points along the x-axis including the boundaries
y = linspace(a, b, N_internal + 2);     % --- Sampling points along the y-axis including the boundaries

[X, Y] = meshgrid(x, y);      
X = X';                                 % --- Transpose so that X(i, j), Y(i, j) are the coordinates of point (i, j)
Y = Y';                      

% --- Interior points
Xint = X(2 : N_internal + 1, 2 : N_internal + 1);       
Yint = Y(2 : N_internal + 1, 2 : N_internal + 1);

% --- rhs values
rhs = f(Xint, Yint);  

% --- Reference solution
uRef = exp(X + Y / 2);  

% --- Include boundary terms into the rhs
rhs(:, 1)               = rhs(:, 1) - uRef(2 : N_internal + 1, 1) / h^2;
rhs(:, N_internal)      = rhs(:, N_internal) - uRef(2 : N_internal + 1, N_internal + 2) / h^2;
rhs(1, :)               = rhs(1, :) - uRef(1, 2 : N_internal + 1) / h^2;
rhs(N_internal, :)      = rhs(N_internal, :) - uRef(N_internal + 2, 2 : N_internal + 1) / h^2;

% --- Set the system matrix
I = speye(N_internal);
e = ones(N_internal, 1);
T = spdiags([e -4 * e e], [-1 0 1], N_internal, N_internal);
S = spdiags([e e], [-1 1], N_internal, N_internal);
A = (kron(I, T) + kron(S, I)) / h^2;

% --- Solve the linear system. Calculate the solution at the only interior
% points
ff = reshape(rhs, N_internal * N_internal, 1);
uSol = A \ ff;
uRefInternal = uRef(2 : N_internal + 1, 2 : N_internal + 1);
uSol = reshape(uSol, N_internal, N_internal);  
err = 100 * sqrt(sum(sum(abs(uSol - uRefInternal).^2)) / sum(sum(abs(uRefInternal).^2)));
fprintf('Percentage rms error = %10.3e \n',err)

% --- Graph
figure(1)
mesh(x(2 : N_internal + 1), y(2 : N_internal + 1), uSol)
