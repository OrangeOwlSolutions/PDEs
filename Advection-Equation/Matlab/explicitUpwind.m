% --- Solves the heat equation u_t + v * u_x = 0 in (0, 2 * pi) 

% --- INPUTS
% --- propagatingFunction           : Solution function
% --- v                             : Wave speed
% --- t0                            : Initial time
% --- tf                            : Final time
% --- M                             : Number of time steps
% --- N                             : Number of space mesh points

% --- OUTPUTS
% --- u                             : Numerical solution
% --- uRef                          : Exact solution
% --- x                             : Space discretization points
% --- t                             : Time discretization points
function [u, uRef, x, t] = explicitUpwind(propagatingFunction, v, t_0, t_f, M, N)

%%%%%%%%%%%%%%%%%%%%%%%%
% SPACE DISCRETIZATION %
%%%%%%%%%%%%%%%%%%%%%%%%
dx  = 2 * pi / N;                   % --- Discretization step in space
x   = (0 : dx : 2 * pi);           % --- Discretization points

%%%%%%%%%%%%%%%%%%%%%%%
% TIME DISCRETIZATION %
%%%%%%%%%%%%%%%%%%%%%%%
dt  = (t_f - t_0) / M;              % --- Discretization time
t   = t_0 : dt : t_f;

alpha = v * dt / dx                % --- Courant number

% --- Initialize the solutions (approximated and exact)
u            = zeros(M + 1, N + 1); % --- u(u, t); First row is for initial condition, first column is for boundary condition
u(1, :)      = propagatingFunction(x - v * t(1));     % --- Initial condition
uRef(1, :)   = u(1, :);

for l = 1 : M       % --- Time steps
    u(l + 1, 2 : N + 1) = u(l, 2 : N + 1) - alpha * (u(l, 2 : N + 1) - u(l, 1 : N)); % --- Update equation
    u(l + 1, 1)         = propagatingFunction(x(1) - v * t(l + 1)); % --- Enforcing boundary condition (left boundary)
    % --- We are aware that we are not at all dealing with right boundary
    % condition. The approach works due to the backward finite difference
    % update.
    uRef(l + 1, :)      = propagatingFunction(x - v * t(l + 1));
end
