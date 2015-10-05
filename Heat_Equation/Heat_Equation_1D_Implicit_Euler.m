clear all
close all
clc

% --- Physical parameters
k               = 0.19;                         % --- Thermal conductivity [W / (m * K)]
rho             = 930.;                         % --- Density [kg / m^3]
cp              = 1340.;                        % --- Specific heat capacity [J / (kg * K)]
alpha           = k / (rho * cp);               % --- Thermal diffusivity [m^2 / s]
len             = 1;                            % --- Total len of the domain [m]
N               = 512;                          % --- Number of grid points
dx              = len / (N - 1);                % --- Discretization step [m]
dt              = dx * dx / (4. * alpha);       % --- Time step [s]
T0              = 0.;                           % --- Temperature at the first end of the domain [C]
T_N_1           = 0.;                           % --- Temperature at the second end of the domain [C]
maxIterNumber   = 100;                          % --- Number of overall time steps
F               = alpha * dt / (dx * dx);       % --- Mesh Fourier number

x = 0 : dx : len;                               % --- Computation grid

% --- Generating the system matrix A
A = zeros(length(x), length(x));
for k = 2 : N - 1
    A(k, k-1) = -F;
    A(k, k+1) = -F;
    A(k, k)   = 1 + 2 * F;
end
A(1, 1) = 1;
A(N, N) = 1;

A = sparse(A);
    
% --- Initial temperature profile
T = (sin(pi * x) + 0.1 * sin(100 * pi * x)).';

% --- Iterations
for t = 1 : maxIterNumber,

    % --- Enforcing the boundary conditions
    T(1) = T0;
    T(N) = T_N_1;
    
    T = A \ T;
   
    % --- Plot figure
    figure(1)
    plot(x, T)
    xlabel('x [m]')
    ylabel('Temperature [°C]')
    title(['Temperature evolution after ', num2str(t * dt),' seconds'])
    drawnow

end    
