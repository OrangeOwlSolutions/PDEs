% --- Derivative of the propagating function

function y = propagatingFunctionDerivative(x)

alpha = (2 * (pi /4)^2);
y = -(2 * x / alpha) .* exp(-x.^2 / alpha);
