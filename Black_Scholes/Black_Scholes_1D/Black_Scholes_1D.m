close all
clear all
clc

% --- Algorithm parameters
K     = 100;                    % --- Strike price
r     = 0.05;                   % --- Continuously compounded risk free interest rat
sigma = 0.2;                    % --- Volatility of the underlying stock
T     = 1;                      % --- Time to expiration

Smax = 200;
J    = 256;
S    = linspace(0, Smax, J)';

% --- Explicit solver
N_explicit = 50000;             % --- Number of time steps for explicit solver
V_explicit = Black_Scholes_Explicit(r, sigma, T, S, max(0, S - K), N_explicit);

subplot(2, 1, 1)
plot(S, V_explicit, 'rx', S, Black_Scholes_Analytical(r, sigma, T, S, K),'-b')
axis([0 Smax 0 max(V_explicit)])
legend('numerical', 'analytic', 'Location', 'NorthWest')
title('Explicit solver')
  
% --- Implicit solver
N_implicit = 2500;              % --- Number of time steps for implicit solver
V_implicit = Black_Scholes_Implicit(r, sigma, T, S, max(0, S - K), N_implicit);

subplot(2, 1, 2)
plot(S, V_implicit, 'rx', S, Black_Scholes_Analytical(r, sigma, T, S, K),'-b')
axis([0 Smax 0 max(V_implicit)])
legend('numerical', 'analytic', 'Location', 'NorthWest')
title('Implicit solver')

