function [NQ, dNQ] = LinearBasisFunctions2D(gaussPoints)

xi = gaussPoints(:, 1); eta = gaussPoints(:, 2); 

% --- Calculate the 4 basis functions at natural coordinates (xi, eta).

NQ = 1/4*[(1 - xi) .* (1 - eta), (1 + xi) .* (1 - eta),...
          (1 + xi) .* (1 + eta), (1 - xi) .* (1 + eta)];    
    
% --- Calculate the derivatives of the 4 basis functions at natural coordinates (xi, eta).
dNQ= 1/4*[-(1 - eta), (1 - eta), (1 + eta), -(1 + eta);
          -(1 - xi), -(1 + xi),  (1 + xi),   (1 - xi)];  
                                                 
