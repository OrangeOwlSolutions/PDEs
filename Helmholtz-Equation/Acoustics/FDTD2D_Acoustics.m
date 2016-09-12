% --- 2D Acoustic Wave Equation using FDTD
clear all
close all
clc

Nx                  = 201;                                                  % --- Number of mesh points along x
Ny                  = 201;                                                  % --- Number of mesh points along y
Lx                  = 200;                                                  % --- Length of the domain along x
Ly                  = 200;                                                  % --- Length of the domain along y
x                   = linspace(0, Lx, Nx);                                  % --- Mesh points along x
y                   = linspace(0, Ly, Ny);                                  % --- Mesh points along y
[X, Y]              = meshgrid(x, y);
dx                  = x(2) - x(1);                                          % --- Mesh step along x
dy                  = y(2) - y(1);                                          % --- Mesh step along y
v                   = 5;                                                    % --- Wave speed
p                   = 0.02;                                                % --- Wave decay factor
dt                  = 0.25 / (v * sqrt((1 / dx)^2 + (1 / dy)^2));           % --- Time-Step matching the Courant-Friedrichs-Lewy condition
T                   = floor((3 * sqrt(Lx^2 + Ly^2) / v) / dt);              % --- Total number of time steps
t                   = (0 : T - 1) * dt;                                     % --- Time axis

% --- Initial conditions
u                   = zeros(Ny, Nx);                                        % --- Current solution u(x, y, t)
indxc               = floor(Nx / 3);                                        % --- Index for the source location along x
indyc               = floor(Ny / 2);                                        % --- Index for the source location along x
xc                  = x(indxc);                                             % --- x-coordinate of source
yc                  = y(indyc);                                             % --- y-coordinate of source
indRc               = 50;
Rc                  = Lx / indRc;
ind = find(sqrt((X - xc).^2 + (Y - yc).^2) <= Rc);                          % --- Source support
u(ind) = exp(-indRc * ((X(ind) - xc).^2 + (Y(ind) - yc).^2) / Lx);
u_old               = u;                                                    % --- Solution at the previous step

c = dt^2 * v^2 / dx^2;                                                      % --- CFL number
q = 2 - p * dt;
r = -1 + p * dt;
steps2plot = 15;
ind1 = 2 : (size(u, 1) - 1);
ind2 = 2 : (size(u, 2) - 1);
u_new = zeros(size(u));
figure(1)
% imagesc(x, y, u); colorbar; caxis([-1 1])
imagesc(x, y, u); colorbar;
for tt = 1 : T,
    u_new(ind1, ind2) = q * u(ind1, ind2) + r * u_old(ind1, ind2) + ...
        c * (u(ind1, ind2 - 1) + u(ind1 - 1, ind2) + u(ind1, ind2 + 1) + u(ind1 + 1, ind2) -...
             4. * u(ind1, ind2));
    
    u_old = u;          % --- Curent solution becomes old
    u = u_new;          % --- New solution becomes current

    if mod(tt - 1, steps2plot)==0
%         imagesc(x, y, u); colorbar; caxis([-1 1])
        imagesc(x, y, u); colorbar; 
        drawnow;
    end
    
end


