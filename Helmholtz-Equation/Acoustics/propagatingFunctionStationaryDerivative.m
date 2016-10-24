% --- Propagating function

function y = propagatingFunctionStationaryDerivative(x, x1, x2)

y = -(2 * pi / ((2 / 3) * (x2 - x1))) * sin((2 * pi / ((2 / 3) * (x2 - x1))) * x);
