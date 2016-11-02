% --- Solves the wave equation u_xx = v^2 * u_tt in (t_0, t_f) x (x1, x2) 

% --- INPUTS
% --- v                             : Wave speed
% --- x1                            : Left boundary of the computational
% domain
% --- x2                            : Right boundary of the computational
% domain
% --- t_0                           : Initial time
% --- t_f                           : Final time
% --- M                             : Number of time steps
% --- N                             : Number of space mesh points

% --- OUTPUTS
% --- u                             : Numerical solution
% --- uRef                          : Exact solution
% --- x                             : Space discretization points
% --- t                             : Time discretization points
function [u, uRef, uFW, uBW, x, t] = stationarySolution(v, t_0, t_f, M, x1, x2, N)

%%%%%%%%%%%%%%%%%%%%%%%%
% SPACE DISCRETIZATION %
%%%%%%%%%%%%%%%%%%%%%%%%
dx  = (x2 - x1) / N;                % --- Discretization step in space
x   = (x1 : dx : x2);               % --- Discretization points    

%%%%%%%%%%%%%%%%%%%%%%%
% TIME DISCRETIZATION %
%%%%%%%%%%%%%%%%%%%%%%%
dt  = (t_f - t_0) / M;              % --- Discretization time
t   = t_0 : dt : t_f;

alpha = v * dt / dx                % --- Courant number

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZE THE SOLUTION (APPROXIMATED AND EXACT) %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Initial condition
u               = zeros(M + 1, N + 1); % --- u(u, t); First row is for initial condition, first column is for boundary condition
uRef            = zeros(M + 1, N + 1); % --- Allocating the reference solution
uFW             = zeros(M + 1, N + 1); % --- Allocating the forward propagating part of the reference solution
uBW             = zeros(M + 1, N + 1); % --- Allocating the backward propagating part of the reference solution
u1(1 : N + 1)   = 0.5 * (propagatingFunctionStationary(x - x1 - v * t(1), x1, x2) - ...
                         propagatingFunctionStationary(x - x1 + v * t(1), x1, x2));   % --- It includes also the two boundary conditions        
u(1, 1 : N + 1) = u1(1 : N + 1);
uRef(1, :)      = u1(1 : N + 1);
uFW(1, :)       = 0.5 * propagatingFunctionStationary(x - x1 - v * t(1), x1, x2);
uBW(1, :)       = -0.5 * propagatingFunctionStationary(x - x1 + v * t(1), x1, x2);

% --- First step
u2              = zeros(1, N + 1);
u2(1)           = 0.5 * (propagatingFunctionStationary(- v * t(2), x1, x2) - propagatingFunctionStationary(v * t(2), x1, x2));  % --- Left boundary condition
u2(2 : N)       = alpha^2 * u1(3 : N + 1) / 2 ...
                + (1 - alpha^2) * u1(2 : N)   ...
                +  alpha^2      * u1(1 : N - 1) / 2 ...
                -  v * dt       * 0.5 * (propagatingFunctionStationaryDerivative(x(2 : N) - x1 - v * t(1), x1, x2) + ...
                                         propagatingFunctionStationaryDerivative(x(2 : N) - x1 + v * t(1), x1, x2));
u2(N + 1)       = 0.5 * (propagatingFunctionStationary(x2 - x1 - v * t(2), x1, x2) - propagatingFunctionStationary(x2 - x1 + v * t(2), x1, x2));  % --- Right boundary condition
u(2, 1 : N + 1) = u2(1 : N + 1);
uRef(2, 1 : N + 1)  = 0.5 * (propagatingFunctionStationary(x - x1 - v * t(2), x1, x2) - propagatingFunctionStationary(x - x1 + v * t(2), x1, x2));  % --- Right boundary condition
uFW(2, :)       = 0.5 * propagatingFunctionStationary(x - x1 - v * t(2), x1, x2);
uBW(2, :)       = -0.5 * propagatingFunctionStationary(x - x1 + v * t(2), x1, x2);

%  Take all the other steps.
%
for l = 2 : M       % --- Time steps
    u3              = zeros(1, N + 1);
    u3(1)           = 0.5 * (propagatingFunctionStationary(- v * t(l + 1), x1, x2) - propagatingFunctionStationary(v * t(l + 1), x1, x2));   % --- Left boundary condition
    u3(2 : N)       = alpha^2 * u2(3 : N + 1) ...               % --- Update equation
                      + 2 * (1 - alpha^2) * u2(2 : N) ...
                      +      alpha^2      * u2(1 : N - 1) ...
                      -                     u1(2 : N);
    u3(N + 1)       = 0.5 * (propagatingFunctionStationary(x2 - x1 - v * t(l + 1), x1, x2) - propagatingFunctionStationary(x2 - x1 + v * t(l + 1), x1, x2));   % --- Right boundary condition

    u(l + 1, 1 : N + 1) ...        
                    = u3(1 : N + 1);
    u1(1 : N + 1)   = u2(1 : N + 1);
    u2(1 : N + 1)   = u3(1 : N + 1);
    
    uRef(l + 1, 1 : N + 1)  = 0.5 * (propagatingFunctionStationary(x - x1 - v * t(l + 1), x1, x2) - propagatingFunctionStationary(x - x1 + v * t(l + 1), x1, x2));  % --- Right boundary condition
    uFW(l + 1, :)       = 0.5 * propagatingFunctionStationary(x - x1 - v * t(l + 1), x1, x2);
    uBW(l + 1, :)       = -0.5 * propagatingFunctionStationary(x - x1 + v * t(l + 1), x1, x2);
end

