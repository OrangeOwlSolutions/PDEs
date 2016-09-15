% --- Method of moments - One point quadrature - Scattering by a circular
% cylinder

close all 
clear all
clc

epsilon0    = 8.85418781 * 1e-12;                                           % --- Free-space dielectric permittivity
mu0         = 4 * pi * 1e-7;                                                % --- Free-space magnetic permeability
zeta0       = sqrt(mu0 / epsilon0);                                         % --- Free-space wave impendance 
c           = 1 / sqrt(epsilon0 * mu0);                                     % --- Wavespeed in vacuum

f           = 3e9;                                                          % --- Operating frequency
omega       = 2 * pi * f;                                                   % --- Operating angular frequency
beta        = omega / c;                                                    % --- Wavenumber
lambda      = 2 * pi / beta;                                                % --- Wavelength

a           = 4 * lambda;                                                             % --- Cylinder radius  

NN          = ceil(2 * pi * a / (lambda / 10));                             % --- Number of discretization intervals of the cylinder's contour
                                            
afar        = 10 * a * a / lambda;                                          % --- Far-field calculation distance

E0 = 1;                                                                     % --- Amplitude of the impinging wave

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DISCRETIZATION OF THE CYLINDER SURFACE %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
t = linspace(0, 2 * pi - 1e-9, NN + 1);

% --- Extremals of the segments discretizing the cylinder surface
x = a * cos(t(1 : NN));
y = a * sin(t(1 : NN));

% --- Matching points on the cylinder surface 
xp(1 : NN - 1) = (x(2 : NN) + x(1 : NN - 1)) / 2;
yp(1 : NN - 1) = (y(2 : NN) + y(1 : NN - 1)) / 2;
tp(1 : NN - 1) = (t(2 : NN) + t(1 : NN - 1)) / 2;
xp(NN) = ((x(NN) + x(1)) / 2);
yp(NN) = ((y(NN) + y(1)) / 2);
tp(NN) = mod(((t(1) + (t(NN) - 2 * pi)) / 2), 2 * pi);        

Deltal = 2 * a * tan(pi / (NN + 1));

%%%%%%%%%%%%%%%%%%%%%%%
% COEFFICIENTS MATRIX %
%%%%%%%%%%%%%%%%%%%%%%%
Xp = ones(NN, 1) * xp - (xp.') * ones(1, NN);
Yp = ones(NN, 1) * yp - (yp.') * ones(1, NN);
A = (omega * mu0 / 4) * besselh(0, 2, beta * sqrt(Xp.^2 + Yp.^2)) * Deltal;

gamma = 1.781072;
A(eye(size(A)) ~= 0) = (omega * mu0 / 4) * Deltal * (-2 * 1i / pi) * (log(gamma * beta * Deltal / 4) + 1i * (pi / 2) - 1);

%%%%%%%%%%%%%%%%%%%
% IMPINGING FIELD %
%%%%%%%%%%%%%%%%%%%
Ei = E0 * exp(-1i * beta * xp.');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SOLUTION - CURRENT RECONSTRUCTION %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
J = linsolve(A, Ei);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SCATTERED FAR-FIELD CALCULATION %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Esfar = zeros(size(tp));
for k = 1 : length(tp),
    Esfar(k) = - (omega * mu0 / 4) * sqrt(2 / (pi * afar)) * Deltal * exp(-1i * beta * afar) * sum(exp(1i * beta * (a * cos(tp - tp(k)))) .* J.');
end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DISCRETIZATION OF THE EXTERIOR REGION %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ae          = 3 * a;                                                        % --- Semi-extent of "exterior" region along x-axis
be          = 3 * a;                                                        % --- Semi-extent of "exterior" region along y-axis
Nx          = ceil(ae / (lambda / 8));                                      % --- Number of discretization points of the "exterior" region along x
Ny          = ceil(ae / (lambda / 8));                                      % --- Number of discretization points of the "exterior" region along y
[Xe, Ye]    = meshgrid(linspace(-ae, ae, Nx), linspace(-be, be, Ny));
indinterior = find(sqrt(Xe.^2 + Ye.^2) <= a);

Es           = zeros(Nx, Ny);                                               % --- Scattered field

for k = 1 : NN,
    
    Es = Es - (omega * mu0 / 4) * Deltal * besselh(0, 2, beta * sqrt((Xe - xp(k)).^2 + (Ye - yp(k)).^2)) * J(k);
    
end
Es(indinterior) = 0;

E = Es + E0 * exp(-1i * beta * Xe);                                         % --- Total field

%%%%%%%%%
% PLOTS %
%%%%%%%%%
figure(1);
plot(tp, abs(J), 'k');
title('Amplitude of the induced current density')

figure(2);
plot(tp, abs(Esfar), 'k');
title('Amplitude of the far-field')

figure(3)
surf(Xe, Ye, abs(Es)); 
view(2), axis([-ae ae -be be]), axis equal, shading interp, colorbar
title('Amplitude of the scattered field')

figure(4)
surf(Xe, Ye, abs(E)); 
view(2), axis([-ae ae -be be]), axis equal, shading interp, colorbar
title('Amplitude of the total field')

%%%%%%%%%%%%%
% ANIMATION %
%%%%%%%%%%%%%
dt = 1 / (8 * f);
kk = 1;
omegat = linspace(0, 2 * pi, ceil((1 / f) / dt));
count = 0;
figure(5)
while kk == 1,
	
    figure(5)
    surf(Xe, Ye, 2 * real(E * exp(1i * omegat(count + 1))));
    view(2), axis([-ae ae -be be]), axis equal, shading interp, colorbar
	drawnow
    
    count = mod(count + 1, length(omegat));
    
end

