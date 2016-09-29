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
function [u, uRef, x, t] = explicitDownwind(propagatingFunction, v, t_0, t_f, M, N)

%%%%%%%%%%%%%%%%%%%%%%%%
% SPACE DISCRETIZATION %
%%%%%%%%%%%%%%%%%%%%%%%%
dx  = 2 * pi / N;                   % --- Discretization step in space
x   = (0 : dx : 2 * pi)';           % --- Discretization points

%%%%%%%%%%%%%%%%%%%%%%%
% TIME DISCRETIZATION %
%%%%%%%%%%%%%%%%%%%%%%%
dt  = (t_f - t_0) / M;              % --- Discretization time
t   = t_0 : dt : t_f;

alpha = v * dt / dx;                % --- Courant number

% --- Initialize the solutions (approximated and exact)
u            = zeros(N + 1, M + 1); % --- u(u, t)
u(:, 1)      = propagatingFunction(x - v * t(1));     % --- Initial condition
uRef(:, 1)   = u(:, 1);

for l = 1 : M       % --- Time step
    u(1 : N, l + 1) = u(1 : N, l) - alpha * (u(2 : N + 1, l) - u(1 : N, l));
    u(1, l + 1)     = propagatingFunction(x(1) - v * t(l + 1));
    uRef(:, l + 1)  = propagatingFunction(x - v * t(l + 1));
end
