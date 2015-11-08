%*********************************************%
%* EXPLICIT APPROACH TO THE BLACK-SCHOLES PDE %
%*********************************************%
function V = Black_Scholes_Explicit(r, sigma, T, S, V, N)

dS     = S(2) - S(1);
dt     = T / N;
lambda = 0.5 * dt * sigma^2 * S.^2 / dS^2;
gamma  = 0.5 * dt * r * S / dS;

% --- Tridiagonal matrix elements
a =         lambda - gamma;
b = 1 - 2 * lambda - r * dt;
c =         lambda + gamma;

% --- Boundary condition d^2u / dS^2 = 0
a(end) =   - 2*gamma(end);
b(end) = 1 + 2*gamma(end) - r*dt;
c(end) = 0;

% --- Construct tridiagonal matrix
A = spdiags([c b a],[-1 0 1],length(a),length(a))';

% --- Time steps
for n = 1 : N
  V = A * V;
end
