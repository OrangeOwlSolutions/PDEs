clear all
close all
clc

%%%%%%%%%%%%%%
% PARAMETERS %
%%%%%%%%%%%%%%

d           = 8*10^-2;                              % --- Length of the investigation domain
rho0        = 1*10^-8;                              % --- Charge density
epsr        = 1.0;                                  % --- Relative dielectric permittivity
eps         = epsr * 8.85*10^-12;                   % --- Dielectric permittivity
Va          = 1;                                    % --- Boundary condition: voltage value at the leftmost node
Vb          = 0;                                    % --- Boundary condition: voltage value at the rightmost node
Ne          = 8;                                    % --- Number of elements
Nn          = Ne + 1;                               % --- Number of nodes
numElementInterpolationPoints = 50;                 % --- Number of interpolation points inside each element

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ELEMENT CONNECTIVITY MATRIX %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for k = 1 : Ne
    elementConnectivityMatrix(k, 1) = k;
    elementConnectivityMatrix(k, 2) = k + 1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ELEMENT COEFFICIENT MATRIX AND ELEMENT RIGHT-HAND-SIDE VECTOR %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
le = d / Ne;                                        % --- Element length
Ke(1, 1) =  eps / le;
Ke(1, 2) = -eps / le;
Ke(2, 1) = -eps / le;
Ke(2, 2) =  eps / le;

fe       = -le * rho0 / 2 * ([1; 1]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GLOBAL COEFFICIENT MATRIX %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K = zeros(Nn);
f = zeros(Nn, 1);
for e = 1 : Ne
    for k = 1 : 2
        for l = 1 : 2
            K(elementConnectivityMatrix(e, k), elementConnectivityMatrix(e, l)) = K(elementConnectivityMatrix(e, k), elementConnectivityMatrix(e, l)) + Ke(k, l);
        end
        f(elementConnectivityMatrix(e, k)) = f(elementConnectivityMatrix(e, k)) + fe(k);
    end
end
for k = 2 : Nn
    f(k) = f(k) - K(k, 1) * Va;
end

K(:, 1)     = 0;
K(1, :)     = 0;
K(1, 1)     = 1;
K(:, Nn)    = 0;
K(Nn, :)    = 0;
K(Nn, Nn)   = 1;

f(1)        = Va;
for k = 2 : Nn - 1
    f(k) = f(k) - K(k, Nn) * Vb;
end
f(Nn)       = Vb;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SOLUTION OF THE LINEAR SYSTEM %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
V = K \ f;

%%%%%%%%%%%%%%%%%
% INTERPOLATION %
%%%%%%%%%%%%%%%%%
for e = 1 : Ne
    x(e, 1) = (e - 1) * le;                     % --- Elements leftmost coordinates
    x(e, 2) = e * le;                           % --- Elements rightmost coordinates
end

dx = le / (numElementInterpolationPoints);      % --- Element discretization step
for e = 1 : Ne
    for k = 1 : numElementInterpolationPoints
        idx = (e - 1) * numElementInterpolationPoints + k;
        xV(idx)             = (idx - 1) * dx;   % --- Voltage abscissas
        xE(idx + e - 1)     = (idx - 1) * dx;   % --- Electric field abscissas
        Vinterp(idx) = V(e) * (x(e, 2) - xV(idx)) / le + V(e + 1) * (xV(idx) - x(e, 1)) / le; % --- Voltage
        Einterp(idx + e - 1)      = (V(e) - V(e + 1)) / le;
    end
    if e < Ne
        xE(idx + e) = xE(idx + e - 1);
        Einterp(idx + e)  = (V(e + 1) - V(e + 2)) / le;   % --- Electric field
    end
end
xV(idx + 1)     = idx * dx;
xE(idx + e)     = idx * dx;
Vinterp(idx + 1)  = V(Ne + 1);
Einterp(idx + e)      = Einterp(idx + e - 1);

%%%%%%%%%%%%%%%%%%%%%%%
% ANALYTICAL SOLUTION %
%%%%%%%%%%%%%%%%%%%%%%%
for k = 1 : Ne * numElementInterpolationPoints + 1
    Vexact(k) = rho0 / (2 * eps) * xV(k)^2 + ((Vb - Va) / d - rho0 * d / (2 * eps)) * xV(k) + Va;
    Eexact(k) = rho0 * (d - 2 * xV(k)) / (2 * eps) + (Va - Vb) / d;
end

%%%%%%%%%
% PLOTS %
%%%%%%%%%
figure(1)
plot(xV, Vinterp, 'k--');   
hold
plot(xV, Vexact,'k-'); 
xlabel('x (meters)');
ylabel('V (Volts)');
legend('FEM', 'Exact');

figure(2);
plot(xE, Einterp, 'k--'); 
hold
plot(xV, Eexact, 'k-'); 
xlabel('x (meters)');
ylabel('Einterp (V/m)');
legend('FEM', 'Exact');

100 * sum(sum(abs(Vinterp - Vexact).^2)) / sum(sum(abs(Vexact).^2))
