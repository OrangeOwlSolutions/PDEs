% --- 1D Acoustic Wave Equation using FDTD
close all
clear all
clc

Nx      = 301;                                                                      % --- Number of mesh points
L       = 2.5;                                                                      % --- Length of the string
x       = linspace(0, L, Nx);                                                       % --- Mesh points
dx      = x(2) - x(1);                                                              % --- Mesh step
v       = 5;                                                                        % --- Wave speed
dt      = 0.25 * dx / v;                                                            % --- Time-Step matching the Courant-Friedrichs-Lewy condition
T       = floor((3 * L / v) / dt);                                                  % --- Total number of time steps
t       = (0 : T - 1) * dt;                                                         % --- Time axis

u   = zeros(T,Nx);                                                                  % --- u(x,t)

% --- Initial conditions
kx = 2;                                                                             % --- Spatial frequency of source
nmax = floor(1 / (2 * kx * dx));  
u(1, 1 : nmax) = sin(2 * pi * kx * x(1 : nmax));
u(2, 1 : nmax) = sin(2 * pi * kx * x(1 : nmax));

%%%%%%%%%%%%%%
% ITERATIONS %
%%%%%%%%%%%%%%
c = v * (dt / dx);                                                                  % --- Courant-Friedrichs-Lewy number
for nt = 3 : T
    for nx = 2 : Nx - 1
        u1          = 2 * u(nt - 1, nx) - u(nt - 2, nx);
        u2          = u(nt - 1, nx - 1) - 2 * u(nt - 1, nx) + u(nt - 1, nx + 1);
        u(nt, nx)   = u1 + c * c * u2;    
    end                   
end

%%%%%%%%%%%%%%%%%%%%%%%
% RECORDING THE MOVIE %
%%%%%%%%%%%%%%%%%%%%%%%
for nt = 1 : T
    plot(x, u(nt, :), 'linewidth', 2);
    grid on;
    axis([min(x) max(x) -2 2]);
    xlabel('x axis', 'fontSize', 14);
    ylabel('Wave amplitude', 'fontSize', 14);              
    titlestring = ['Time step = ',num2str(nt), ' Time = ',num2str(t(nt)), 'seconds'];
    title(titlestring, 'fontsize', 14);                            
    h = gca; 
    get(h, 'FontSize') 
    set(h, 'FontSize', 14);
    fh = figure(5);
    set(fh, 'color', 'white'); 
    F = getframe;            
end

movie(F, T, 1)
