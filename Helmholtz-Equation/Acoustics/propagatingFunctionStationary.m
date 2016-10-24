% --- Propagating function

function y = propagatingFunctionStationary(x, x1, x2)

y = cos((2 * pi / ((2 / 3) * (x2 - x1))) * x);
