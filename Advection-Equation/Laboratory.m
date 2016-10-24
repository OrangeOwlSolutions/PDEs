clear all
close all
clc

t_0 = 0;                                    % --- Initial time
t_f = 15;                                   % --- Final time
M   = 200;                                  % --- Number of time steps
N   = 165;                                  % --- Number of space mesh points
v   = 0.5;                                  % --- Wave speed

% --- The numerical solution is arranged in a (M + 1) x (N + 1) matrix.
% Space runs along the columns, while time runs along the rows.
% The first row hosts the initial condition, while the first column hosts
% the left boundary condition.

% --- Try first forward difference discretization with positive
% perturbation speed to see that downwind updates fail
% [uDownwind, uRef, x, t] = explicitDownwind(@propagatingFunction, v, t_0, t_f, M, N);
% for m = 1 : M
%     plot(x, uRef(m, :), 'r', 'LineWidth', 2);
%     hold on
%     plot(x, uDownwind(m, :), 'LineWidth', 2);
%     axis([0, 2 * pi, -.3, 1.3]);
%     title('Explicit Downwind', 'FontSize', 14)
%     xlabel('Red: reference solution; Blue: numerical solution', 'FontSize', 14)
%     hold off
%     figure(1)
%     pause(0.1)
% end
% 
% pause

% --- Try then backward difference discretization with positive
% perturbation speed to see that upwind updates work correctly
% --- Try also the case when M = 50: Courant condition is not met
% --- Try also N = 160: negligible attenuation case
[uUpwind, uRef, x, t] = explicitUpwind(@propagatingFunction, v, t_0, t_f, M, N);
for m = 1 : M
    plot(x, uRef(m, :), 'r', 'LineWidth', 2);
    hold on
    plot(x, uUpwind(m, :), 'LineWidth', 2);
    axis([0, 2 * pi, -.3, 1.3]);
    title('Explicit Downwind', 'FontSize', 14)
    xlabel('Red: reference solution; Blue: numerical solution', 'FontSize', 14)
    hold off
    figure(1)
    pause(0.1)
end

pause

% --- Try this to see that second order approximation of space derivative leads
% to an unstable approach
% [uCenteredDifference, uRef, x, t] = centeredDifference(@propagatingFunction, v, t_0, t_f, M, N);
% for m = 1 : M
%     plot(x, uRef(m, :), 'r', 'LineWidth', 2);
%     hold on
%     plot(x, uCenteredDifference(m, :), 'LineWidth', 2);
%     axis([0, 2 * pi, -.3, 1.3]);
%     title('Centered Difference', 'FontSize', 14)
%     xlabel('Red: reference solution; Blue: numerical solution', 'FontSize', 14)
%     hold off
%     figure(1)
%     pause(0.1)
% end
% 
% pause

% --- Try this to see that Lax Friedrichs requires proper treatment of the
% right boundary
% [uLaxFriedrichsNoRightBoundary, uRef, x, t] = laxFriedrichsNoRightBoundary(@propagatingFunction, v, t_0, t_f, M, N);
% for m = 1 : M
%     plot(x, uRef(m, :), 'r', 'LineWidth', 2);
%     hold on
%     plot(x, uLaxFriedrichsNoRightBoundary(m, :), 'LineWidth', 2);
%     axis([0, 2 * pi, -.3, 1.3]);
%     title('Lax Friedrichs no right boundary', 'FontSize', 14)
%     xlabel('Red: reference solution; Blue: numerical solution', 'FontSize', 14)
%     hold off
%     figure(1)
%     pause(0.1)
% end
% 
% pause

% --- Lax Friedrichs with treatment of the right boundary
% [uLaxFriedrichs, uRef, x, t] = laxFriedrichs(@propagatingFunction, v, t_0, t_f, M, N);
% for m = 1 : M
%     plot(x, uRef(m, :), 'r', 'LineWidth', 2);
%     hold on
%     plot(x, uLaxFriedrichs(m, :), 'LineWidth', 2);
%     axis([0, 2 * pi, -.3, 1.3]);
%     title('Lax Friedrichs', 'FontSize', 14)
%     xlabel('Red: reference solution; Blue: numerical solution', 'FontSize', 14)
%     hold off
%     figure(1)
%     pause(0.1)
% end
% 
% pause

% --- Lax Wendroff
% [uLaxWendroff, uRef, x, t] = laxWendroff(@propagatingFunction, v, t_0, t_f, M, N);
% for m = 1 : M
%     plot(x, uRef(m, :), 'r', 'LineWidth', 2);
%     hold on
%     plot(x, uLaxWendroff(m, :), 'LineWidth', 2);
%     axis([0, 2 * pi, -.3, 1.3]);
%     title('Lax Wendroff', 'FontSize', 14)
%     xlabel('Red: reference solution; Blue: numerical solution', 'FontSize', 14)
%     hold off
%     figure(1)
%     pause(0.1)
% end
% 
% pause

% --- Leapfrog
% [uLeapFrog, uRef, x, t] = leapFrog(@propagatingFunction, v, t_0, t_f, M, N);
% for m = 1 : M
%     plot(x, uRef(m, :), 'r', 'LineWidth', 2);
%     hold on
%     plot(x, uLeapFrog(m, :), 'LineWidth', 2);
%     axis([0, 2 * pi, -.3, 1.3]);
%     title('Leapfrog', 'FontSize', 14)
%     xlabel('Red: reference solution; Blue: numerical solution', 'FontSize', 14)
%     hold off
%     figure(1)
%     pause(0.1)
% end
% 
% pause

% --- Final comparison
% for m = 1 : M
%     plot(x, uRef(m, :), 'r*', 'LineWidth', 2);
%     hold on
%     plot(x, uUpwind(m, :),       'LineWidth', 2);
%     plot(x, uLaxFriedrichs(m, :),  'g', 'LineWidth', 2);
%     plot(x, uLaxWendroff(m, :),  'k', 'LineWidth', 2);
%     plot(x, uLeapFrog(m, :), 'c', 'LineWidth', 2);
%     axis([0, 2 * pi, -.3, 1.3]);
%     legend('Reference', 'Explicit Upwind', 'Lax-Friedrichs', 'Lax Wendroff', 'LeapFrog')
%     xlabel('x', 'FontSize', 14)
%     hold off
%     figure(1)
%     pause(0.1)
% end

