% --- Solves the 1D heat equation with an explicit finite difference scheme

clear all
close all
clc

% --- Physical parameters
k               = 0.19;                         % --- Thermal conductivity [W / (m * K)]
rho             = 930.;                         % --- Density [kg / m^3]
cp              = 1340.;                        % --- Specific heat capacity [J / (kg * K)]
alpha           = k / (rho * cp);               % --- Thermal diffusivity [m^2 / s]
length          = 1.6;                          % --- Total length of the domain [m]
N               = 32768;                        % --- Number of grid points
dx              = length / (N - 1);             % --- Discretization step [m]
dt              = dx * dx / (4. * alpha);       % --- Time step [s]
T0              = 0.;                           % --- Temperature at the first end of the domain [C]
Q_N_1           = 10.;                          % --- Heat flux at the second end of the domain [W / m^2]
maxErr          = 1.0e-5;                       % --- Maximum admitted DeltaT
maxIterNumber   = 10.0 / dt;                    % --- Number of overall time steps

x = 0 : dx : length;                            % --- Computation grid

% --- Initial temperature profile
T = zeros(size(x)) + T0;

% --- Iteration counter
count = 0;

% --- "Temperature error" DeltaT
DeltaT = zeros(size(x));

% --- Maximum "temperature error" DeltaT
maxDeltaT = Inf;

while ((count <= maxIterNumber) && (maxDeltaT > maxErr))

    % --- Internal region between the two boundaries.
    DeltaT(2 : N - 1) = dt * alpha * ((T(1 : N - 2) + T(3 : N) - 2. * T(2 : N - 1)) / (dx * dx));
    
    % --- Enforcing boundary condition at the right end.
    DeltaT(N) = dt * 2. * ((k * ((T(N - 1) - T(N)) / dx) + Q_N_1) / (dx * rho * cp));
    
	% --- Update the temperature and find the maximum DeltaT over all nodes
 	T(2 : N) = T(2 : N) + DeltaT(2 : N); 
    maxDeltaT = max(abs(DeltaT));
    
    count = count + 1;
    
    % --- Plot figure
    figure(1)
    plot(x(x > 1.592), T(x > 1.592))
    axis([1.592 1.6 0 0.08])
    xlabel('x [m]')
    ylabel('Temperature [°C]')
    title(['Temperature evolution after ', num2str(count * dt),' seconds'])
    drawnow

end    
    
