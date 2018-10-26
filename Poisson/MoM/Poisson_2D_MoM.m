% --- Sadiku, Elements of Electromagnetics, Chapter 15
clear all
close all
clc

epsilon0    = 8.8541e-12;                       % --- Free-space dielectric permittivity
a           = 1.0;                              % --- Semi-extent of the plate along x
b           = 1.0;                              % --- Semi-extent of the plate along y
d           = 1.0;                              % --- Distance between the plates
Nx          = 16;                               % --- Number of cells along x
Ny          = 16;                               % --- Number of cells along y
dx          = a / Nx;                           % --- Cell size along x
dy          = b / Ny;                           % --- Cell size along y

% --- Plates discretization
x           = dx * ((1 : Nx) - 0.5);            % --- Cell centers along x
y           = dy * ((1 : Ny) - 0.5);            % --- Cell centers along y
[X, Y]      = meshgrid(x, y);

% --- Plate voltages
V(1 : Nx * Ny) = 1;                             % --- Upper plate
V(Nx * Ny + 1 : 2 * Nx * Ny) = -1;              % --- Lower plate

% --- System matrix
A = zeros(2 * Nx * Ny, 2 * Nx * Ny);
% --- Upper plate
for k = 1 : Nx * Ny,
    % --- Upper plate contribution
    Rij = reshape(sqrt((X - X(k)).^2 + (Y - Y(k)).^2), 1, Nx * Ny);
    A(k, 1 : Nx * Ny) = dx * dy ./ (4 * pi * epsilon0 * Rij);
    indices = find(Rij == 0);
    A(k, indices) = dx * log(1 + sqrt(2)) / (pi * epsilon0);
    % --- Lower plate contribution
    Rij = reshape(sqrt((X - X(k)).^2 + (Y - Y(k)).^2 + d^2), 1, Nx * Ny);
    A(k, Nx * Ny + 1 : 2 * Nx * Ny) = dx * dy ./ (4 * pi * epsilon0 * Rij);
end
% --- Lower plate
for k = Nx * Ny + 1 : 2 * Nx * Ny,
    % --- Upper plate contribution
    Rij = reshape(sqrt((X - X(k - Nx * Ny)).^2 + (Y - Y(k - Nx * Ny)).^2 + d^2), 1, Nx * Ny);
    A(k, 1 : Nx * Ny) = dx * dy ./ (4 * pi * epsilon0 * Rij);
    % --- Lower plate contribution
    Rij = reshape(sqrt((X - X(k - Nx * Ny)).^2 + (Y - Y(k - Nx * Ny)).^2), 1, Nx * Ny);
    A(k, Nx * Ny + 1 : 2 * Nx * Ny) = dx * dy ./ (4 * pi * epsilon0 * Rij);
    indices = find(Rij == 0);
    A(k, Nx * Ny + indices) = dx * log(1 + sqrt(2)) / (pi * epsilon0);
end

rho = inv(A) * V.';

rhoUp   = reshape(rho(1 : Nx * Ny), Nx, Ny);
rhoDown = reshape(rho(Nx * Ny + 1 : 2 * Nx * Ny), Nx, Ny);

C = dx * dy * sum(sum(rhoUp)) / 2;

figure(1)
title('Upper plate')
imagesc(x, y, rhoUp), colorbar
axis 'square'

figure(2)
title('Lower plate')
imagesc(x, y, rhoDown), colorbar
axis 'square'
