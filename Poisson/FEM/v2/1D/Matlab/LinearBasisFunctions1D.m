function [Ne, dNe] = LinearBasisFunctions1D(xi);

% --- Calculate the two basis functions at the natural coordinate xi
N1e  = (1 / 2) * (1 - xi);
N2e  = (1 / 2) * (xi + 1);
Ne   = [N1e, N2e];  

% --- Calculate the derivatives of the basis function with respect to natural coordinate xi
dN1e = -1 / 2;
dN2e = 1 / 2;
dNe  = [dN1e, dN2e];                       
