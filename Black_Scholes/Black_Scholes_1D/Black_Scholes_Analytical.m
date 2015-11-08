%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ANALYTICAL SOLUTION TO BLACK-SCHOLES EQUATION %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function V = Black_Scholes_Analytical(r, sigma, T, S , K)

S  = max(1e-100, S);     % --- Avoids numerical problems with S = 0
K  = max(1e-100, K);     % --- Avoids numerical problems with K = 0

d1 = (log(S) - log(K) + (r + 0.5 * sigma^2) * T) / (sigma * sqrt(T));
d2 = (log(S) - log(K) + (r - 0.5 * sigma^2) * T) / (sigma * sqrt(T));

V = S .* CND(d1) - exp(-r*T) * K.* CND(d2);




