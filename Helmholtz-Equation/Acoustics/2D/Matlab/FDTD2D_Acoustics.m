% --- 2D Acoustic Wave Equation using FDTD
clear all
close all
clc

% --- Assumption dx = dy

Nx                  = 512;                                                  % --- Number of mesh points along x
Ny                  = 512;                                                  % --- Number of mesh points along y
Lx                  = 200;                                                  % --- Length of the domain along x
Ly                  = 200;                                                  % --- Length of the domain along y
x                   = linspace(0, Lx, Nx);                                  % --- Mesh points along x
y                   = linspace(0, Ly, Ny);                                  % --- Mesh points along y
[X, Y]              = meshgrid(x, y);
dx                  = x(2) - x(1);                                          % --- Mesh step along x
dy                  = y(2) - y(1);                                          % --- Mesh step along y
v                   = 5;                                                    % --- Wave speed
dt                  = 0.25 / (v * sqrt((1 / dx)^2 + (1 / dy)^2));           % --- Time-Step matching the Courant-Friedrichs-Lewy condition
T                   = floor((3 * sqrt(Lx^2 + Ly^2) / v) / dt);              % --- Total number of time steps
t                   = (0 : T - 1) * dt;                                     % --- Time axis

mask = ones(size(X));
mask(find((abs(X - Lx / 2) <= 5) & ((Y <= 90) | (Y >= 110)))) = 0; 

% --- Initial conditions
u                   = zeros(Ny, Nx);                                        % --- Current solution u(x, y, t)
u_old               = zeros(Ny, Nx);                                        % --- Past solution u_old(x, y, t)
u_new               = zeros(Ny, Nx);                                        % --- New solution u_new(x, y, t(
u_plot              = zeros(Ny, Nx);
indxc               = floor(Nx / 3);                                        % --- Index for the source location along x
indyc               = floor(Ny / 2);                                        % --- Index for the source location along y
xc                  = x(indxc);                                             % --- x-coordinate of source
yc                  = y(indyc);                                             % --- y-coordinate of source
alphaSquared        = dt^2 * v^2 / dx^2;                                    % --- CFL number squared
Rc                  = 4;                                                    % --- Radius of the initial source support
ind = find(sqrt((X - xc).^2 + (Y - yc).^2) <= Rc);                          % --- Source support
ind1 = 2 : (size(u, 1) - 1);
ind2 = 2 : (size(u, 2) - 1);
u_old(ind) = exp(-((X(ind) - xc).^2 + (Y(ind) - yc).^2) / Rc^2);
% --- First step: time derivative at initial time is zero
u(ind1, ind2)        = u_old(ind1, ind2) + 0.5 * alphaSquared * (u(ind1, ind2 - 1) + u(ind1 - 1, ind2) + u(ind1, ind2 + 1) + u(ind1 + 1, ind2) - ...
                       4. * u(ind1, ind2));                                  % --- Solution at the first step
% --- Zeroing the old and new solution on the "wall"
u       = u .* mask;                                                         
u_old   = u_old .* mask;
u_plot  = u;
u_plot(find(1-mask)) = 1000;

steps2plot = 15;
figure(1)
imagesc(x, y, u); colorbar; caxis([-.05 .05])
for tt = 1 : T,
    % --- Update formula
    u_new(ind1, ind2) = 2 * u(ind1, ind2) - u_old(ind1, ind2) + ...
        alphaSquared * (u(ind1, ind2 - 1) + u(ind1 - 1, ind2) + u(ind1, ind2 + 1) + u(ind1 + 1, ind2) - ...
             4. * u(ind1, ind2));
    
    u_old = u;          % --- Current solution becomes old
    u = u_new;          % --- New solution becomes current

    % --- Zeroing the old and new solution on the "wall"
    u = u .* mask;
    u_old = u_old .* mask;
    u_plot  = u;
    u_plot(find(1 - mask)) = 1000;

    if mod(tt - 1, steps2plot)==0
        imagesc(x, y, u_plot); colorbar; caxis([-.05 .05])
        drawnow;
    end
    
end


