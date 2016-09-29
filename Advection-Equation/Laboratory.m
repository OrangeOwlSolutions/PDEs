clear all
close all
clc

t_0 = 0;            % --- Initial time
t_f = 15;           % --- Final time
M   = 200;          % --- Number of time steps
N   = 100;          % --- Number of space mesh points
v   = 0.5;          % --- Wave speed

[u, uRef, x, t] = centeredDifference(@propagatingFunction, v, t_0, t_f, M, N);
for m = 1 : M
    plot(x, uRef(:, m), 'r', 'LineWidth', 2);
    hold on
    plot(x, u(:, m), 'LineWidth', 2);
    axis([0, 2 * pi, -.3, 1.3]);
    title('Centered Difference', 'FontSize', 14)
    xlabel('Red: reference solution; Blue: numerical solution', 'FontSize', 14)
    hold off
    figure(1)
    pause(0.1)
end

pause

[u, uRef, x, t] = explicitDownwind(@propagatingFunction, v, t_0, t_f, M, N);
for m = 1 : M
    plot(x, uRef(:, m), 'r', 'LineWidth', 2);
    hold on
    plot(x, u(:, m), 'LineWidth', 2);
    axis([0, 2 * pi, -.3, 1.3]);
    title('Explicit Downwind', 'FontSize', 14)
    xlabel('Red: reference solution; Blue: numerical solution', 'FontSize', 14)
    hold off
    figure(1)
    pause(0.1)
end

pause

[uEU,  uRef, x, t] = explicitUpwind(@propagatingFunction, v, t_0, t_f, M, N);
[uLF,  uRef, x, t] = laxFriedrichs(@propagatingFunction, v, t_0, t_f, M, N);
[uLW,  uRef, x, t] = laxWendroff(@propagatingFunction,v, t_0,t_f,M,N);
[uLFr, uRef, x, t] = leapFrog(@propagatingFunction, v, t_0, t_f, M, N);
for m = 1 : M
    plot(x, uRef(:, m), 'r*', 'LineWidth', 2);
    hold on
    plot(x, uEU(:, m),       'LineWidth', 2);
    plot(x, uLF(:, m),  'g', 'LineWidth', 2);
    plot(x, uLW(:, m),  'k', 'LineWidth', 2);
    plot(x, uLFr(:, m), 'c', 'LineWidth', 2);
    axis([0, 2 * pi, -.3, 1.3]);
    legend('Reference', 'Explicit Upwind', 'Lax-Friedrichs', 'Lax Wendroff', 'LeapFrog')
    xlabel('x', 'FontSize', 14)
    hold off
    figure(1)
    pause(0.1)
end

