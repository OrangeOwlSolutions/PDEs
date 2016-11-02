clear all
close all
clc

t_0 = 0;                                    % --- Initial time
t_f = 15;                                   % --- Final time
x1  = 0;                                    % --- Left boundary of the computational domain
x2  = 2 * pi;                               % --- Right boundary of the computational domain
M   = 200;                                  % --- Number of time steps
N   = 100;                                  % --- Number of space mesh points
v   = 0.5;                                  % --- Wave speed

% --- The numerical solution is arranged in a (M + 1) x (N + 1) matrix.
% Space runs along the columns, while time runs along the rows.
% The first row hosts the initial condition, while the first column hosts
% the left boundary condition.

% [uTraveling, uRef, x, t] = travelingSolution(v, t_0, t_f, M, x1, x2, N);
% for m = 1 : M
%     plot(x, uRef(m, :), 'r', 'LineWidth', 2);
%     hold on
%     plot(x, uTraveling(m, :), '*', 'LineWidth', 2);
%     axis([0, 2 * pi, -0.2, 1.1]);
%     title('Traveling wave', 'FontSize', 14)
%     xlabel('Red: reference solution; Blue: numerical solution', 'FontSize', 14)
%     hold off
%     figure(1)
%     pause(0.1)
% end
% 
% pause

[uStationary, uRef, uFW, uBW, x, t] = stationarySolution(v, t_0, t_f, M, x1, x2, N);
uStationary = reshape(load('D:\Project\FDTD\FDTD_1D_Acoustics\FDTD_1D_Acoustics_CUDA\FDTD_1D_Acoustics_CUDA\d_u.txt'), N + 1, M + 1).'; 
for m = 1 : M
    plot(x, uRef(m, :), 'r', 'LineWidth', 2);
    hold on
    plot(x, uFW(m, :), 'g', 'LineWidth', 2);
    plot(x, uBW(m, :), 'c', 'LineWidth', 2);
    plot(x, uStationary(m, :), '*', 'LineWidth', 2);
    axis([0, 2 * pi, -1.1, 1.1]);
    title('Stationary wave', 'FontSize', 14)
    xlabel('Red: reference solution; Blue: numerical solution', 'FontSize', 14)
    hold off
    figure(1)
    pause(0.1)
end

pause

