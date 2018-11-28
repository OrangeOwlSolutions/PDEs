close all; 
clear all; 
clc;

% --- Resolution domain
a = -1;                                             % --- Left border of the resolution domain
b = 1;                                              % --- Right border of the resolution domain

% --- Dirichlet boundary conditions
Ta = 0.1;                                           % --- Boundary conditions at the left border of the resolution domain
Tb = 0.1;                                           % --- Boundary conditions at the right border of the resolution domain

Nintervals  = 100;                                  % --- Number of discretization intervals
Ninner      = Nintervals - 1;                       % --- Number of inner mesh nodes
Deltax      = (b - a) / Nintervals;                 % --- Discretization step
x           = a : Deltax : b;                       % --- Node coordinates

% --- Right-hand side function
alpha       = -log(Ta) / b^2;
% f           = @(x)(40 * (x - 0.9).^2);
f           = @(x)(2 * alpha * exp(-alpha * x.^2) .* (1 - 2 * alpha * x.^2));
    
% --- Exact solution
Tref        = @(x)(exp(-alpha * x.^2));

% --- Linear system matrix
d           = ones(Ninner, 1);
A           = spdiags([-d , 2 * d , -d], -1 : 1, Ninner, Ninner) / (Deltax^2);

% --- Right-hand side vector
brhs        = feval(f, x(2 : (end - 1)));
brhs(1)     = brhs(1)   + Ta / (Deltax^2);
brhs(end)   = brhs(end) + Tb / (Deltax^2);

% --- Solve the linear system
ysol        = A \ brhs.';

% --- Analytical solution
yref        = feval(Tref, x(2 : (end - 1))); 

% --- Percentage rms error
error       = 100 * sqrt(sum(abs(ysol.' - yref).^2) / sum(abs(yref).^2)); 
fprintf('Percentage rms error %f\n', error);

% --- Plot    
figure(1)
plot(x(2 : (end - 1)), ysol, 'b'); hold on;
plot(x(2 : (end - 1)), yref, 'ro');
legend('Numerical solution', 'Analytical solution');
xlabel('x'),
ylabel('T');
hold off       

