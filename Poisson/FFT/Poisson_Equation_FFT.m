clear all
close all
clc

M       = 32;                                               % --- Number of Fourier harmonics along x (should be a multiple of 2)  
N       = 128;                                              % --- Number of Fourier harmonics along y (should be a multiple of 2)  
Lx      = 3;                                                % --- Domain size along x
Ly      = 1.5;                                              % --- Domain size along y
sigma   = 0.1;                                              % --- Characteristic width of f (make << 1)

% --- Wavenumbers
kx = (2 * pi / Lx) * [0 : (M / 2 - 1) (- M / 2) : (-1)];    % --- Wavenumbers along x
ky = (2 * pi / Ly) * [0 : (N / 2 - 1) (- N / 2) : (-1)];    % --- Wavenumbers along y
[Kx, Ky]  = meshgrid(kx, ky); 

% --- Discretization
hx              = Lx / M;                                   % --- Grid spacing along x
hy              = Ly / N;                                   % --- Grid spacing along y
x               = (0 : (M - 1)) * hx;
y               = (0 : (N - 1)) * hy;
[X, Y]          = meshgrid(x, y);

% --- Compute the right-hand side of differential equation
rSquared        = (X - 0.5 * Lx).^2 + (Y - 0.5 * Ly).^2;
sigmaSquared    = sigma^2;
f               = exp(-rSquared / (2 * sigmaSquared)) .* (rSquared - 2 * sigmaSquared) / (sigmaSquared^2);

% --- Compute forward FFT of right-hand side
fHat            = fft2(f);

% --- Denominator of the unknown spectrum
den             = -(Kx.^2 + Ky.^2); 
den(1, 1)       = 1;                                        % --- Avoid division by zero at wavenumber (0, 0)

% --- Solve Poisson equation in Fourier space 
UHat            = ifft2(fHat ./ den);

% --- Unknown determination
U               = real(UHat);
U               = U - U(1,1);                               % --- Force arbitrary constant to be zero by forcing U(1, 1) = 0

% --- Plots
uRef    = exp(-rSquared / (2 * sigmaSquared));
err     = 100 * sqrt(sum(sum(abs(U - uRef).^2)) / sum(sum(abs(uRef).^2)));
errMax  = norm(U(:) - uRef(:), inf);
fprintf('Percentage root mean square error = %2.15f\n', err);
fprintf('Maximum error = %2.15f\n', errMax);

%%%%%%%%%%
% GRAPHS %
%%%%%%%%%%
figure(1)
surf(X, Y, U, 'EdgeColor', 'none')
shading interp
axis([0 max(x) 0 max(y) min(min(uRef)) max(max(uRef))])
xlabel('x')
ylabel('y')
zlabel('U')
title('Solution of 2D Poisson equation by spectral method')

figure(2)
surf(X, Y, uRef, 'EdgeColor', 'none')
shading interp
axis([0 max(x) 0 max(y) min(min(uRef)) max(max(uRef))])
xlabel('x')
ylabel('y')
zlabel('U')
title('Reference solution')

% --- Loading the CUDA result
load d_result.txt
% --- Transposition is necessary since CUDA writes the matrix row-wise,
% while Matlab reads it column-wise.
d_result = reshape(d_result, M, N).';
