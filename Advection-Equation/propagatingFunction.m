% --- Advective propagating function

function y = propagatingFunction(x)

% y  = x <= (pi-1);
y = exp(-x.^2 / (2 * (pi /4)^2));
