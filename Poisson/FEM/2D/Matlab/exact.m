function disp = exact(x)

% global I

global c E nu P l

I = 2.0/3*c^3;
G = E/(2*(1+nu));

disp(1) = -P * x(1)^2 * x(2) / (2 * E * I) - nu * P * x(2)^3 / (6 * E * I) + P * x(2)^3 / (6 * I * G) + (P * l^2 / (2 * E * I) - P * c^2 / (2 * I * G)) * x(2);
disp(2) = nu * P * x(1) * x(2)^2 / (2 * E * I) + P * x(1)^3 / (6 * E * I) - P * l^2 * x(1) / (2 * E * I) + P * l^3 / (3 * E * I);


