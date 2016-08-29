% --- Solving the 2-D Laplace's equation by the Finite Difference Method 
% --- The exploited numerical scheme is a second order central difference (5-point difference)

clear all
close all
clc

global aP aW aE aS aN b temp L2DifferenceArray                                     

NUM     = 1024;                         % --- Problem size along one size. The computational domain is squared.

TN      = 1.0;							% --- Temperature at top boundary

% --- Number of cells in x and y directions including boundary cells
Ncols   = NUM + 2;                      
Nrows   = NUM + 2;

tol     = 1.e-6;						% --- SOR iteration tolerance
maxIter = 1e3;                         % --- Maximum number of iterations
% maxIter = 200;                         % --- Maximum number of iterations

omega = 1.85;

b           = zeros(NUM, NUM);
b(NUM, :)   = TN;

aW          = ones(NUM, NUM);
aW(:, 1)    = 0;

aE          = ones(NUM, NUM);
aE(:, NUM)  = 0;

aS          = ones(NUM, NUM);
aS(1, :)    = 0;

aN          = ones(NUM, NUM);
aN(NUM, :)  = 0;

aP          = 4 * ones(NUM, NUM);

% --- Initial conditions
temp    = zeros(Nrows, Ncols);                  

%%%%%%%%%%%%%%
% ITERATIONS %
%%%%%%%%%%%%%%
indy = 2 : Ncols - 1;
indx = 2 : Nrows - 1;
L2DifferenceArray   = zeros(NUM + 2, NUM + 2);
for iter = 1 : maxIter

    redUpdate(omega, NUM);
    
	blackUpdate(omega, NUM)
    
    norm_L2 = sqrt(sum(sum(L2DifferenceArray)) / (NUM * NUM));                               
    if (mod(iter - 1, 100) == 0) 
        fprintf('%5d, %0.6f\n', iter - 1, norm_L2);
    end

    if (norm_L2 < tol) 
        break;
    end
end

figure(1)
imagesc(temp(indx, indy)), colorbar
