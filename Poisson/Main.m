% --- Solves the following differential equation
%
%      -d/dx (p(x) du/dx) + q(x)*u  =  f(x)
%
%     by the Finite Element Method (FEM) using linear basis functions.
%     u is an unknown function defined on [x0, x1], and p, q and f are known functions of x.
%
% --- Essential boundary condition is applied at the left end.
% --- Natural   boundary condition is applied at the right end.

clc;
clear all;
close all; 

global totalNumNodes numElements numNodesPerElement totalNumNodes numberGaussPoints globalNodes connectivityMatrix globalIndicesSpecifiedValues specifiedDegreesOfFreedom 
global K f d I J X ntriplets 

x0                      = 0.0;                              % --- Left end of the 1D domain
x1                      = 1.0;                              % --- Right end of the 1D domain

numElements             = 50;                               % --- Total number of elements

totalNumNodes           = numElements + 1;                  % --- Total number of nodes
numNodesPerElement      = 2;                                % --- Number of nodes per element
     
globalNodes             = linspace(x0, x1, totalNumNodes);  % --- Global node coordinates
   
% --- Connectivity matrix
% --- global node number = connectivityMatrix(local node number, element number) 
connectivityMatrix       = zeros(numNodesPerElement, numElements);              
     
for e = 1 : numElements             % --- Loop over all elements, 
    connectivityMatrix(1, e) = e;       
    connectivityMatrix(2, e) = e + 1;   
end                       
                          
numberGaussPoints       = 2;                                % --- Number of Gauss points

f                       = zeros(totalNumNodes, 1);          % --- Global force vector
d                       = zeros(totalNumNodes, 1);          % --- Global solution vector

% --- Triplet for assembling the sparse matrix
I           = zeros(numNodesPerElement * totalNumNodes, 1); % --- Row indices of non-zero entries
J           = zeros(numNodesPerElement * totalNumNodes, 1); % --- Column indices of non-zero entries
X           = zeros(numNodesPerElement * totalNumNodes, 1); % --- Non-zero entries matrix

ntriplets   = 0;                                            % --- Global variable to trace the index

[gaussPoints, gaussWeights] = gaussianQuadrature(numberGaussPoints);     % --- Return Gauss quadrature points and weights                      

%%%%%%%%%%%%
% FEM LOOP %
%%%%%%%%%%%%
for e = 1 : numElements   % --- Loop over the elements
   
    Ke = zeros(numNodesPerElement);                         % --- Element stiffness matrix
    fe = zeros(numNodesPerElement, 1);                      % --- Element force (load) vector 
   
    globalNodeNumbers = connectivityMatrix(:, e);           % --- Global node numbers corresponding to the current element 
    xe = globalNodes(globalNodeNumbers)';                   % --- Global coordinates of the element nodes as a column vector
   
    % --- Calculate the element integral
	for kk = 1 : length(gaussPoints)    % --- Loop over all the Gauss points
       
        [Ne, dNe] = LinearBasisFunctions1D(gaussPoints(kk));% --- Return the 1D linear basis functions along with their derivatives with respect to local coordinates xi at 
                                                            %     Gauss points
                                        
        x   = Ne * xe;                                      % --- Global coordinate (here x) of the current integration point

        Jacobian     = dNe * xe;                            % --- Jacobian dx / dxi

        JacxW = Jacobian * gaussWeights(kk);                % --- Calculate the integration weight

        B  = dNe / Jacobian;                                % --- Calculate the derivatives of the basis functions with respect to x direction 

        Ke = Ke + (B'* pp(x) * B) * JacxW;

        fe = fe + ff(x) * Ne' * JacxW;
      
    end
   
    AssembleGlobalMatrix(e, Ke, fe);     % --- Assemble global matrix
   
end

% --- Constructs the global stiffness matrix in sparse format using I, J, X
%     Any elements of s that have duplicate values of i and j are added together.
%     Accordingly, if there are duplicates (which a finite element matrix always has) the 
%     duplicates are summed, which is exactly what we want when assembling a finite-element 
%     matrix.
K = sparse(I(1:ntriplets), J(1:ntriplets), X(1:ntriplets), totalNumNodes, totalNumNodes);

clear global I J X;
clear Ke fe;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% APPLY BOUNDARY CONDITIONS %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Essential boundary condition (u specified) at first end
globalIndicesSpecifiedValues = 1;   % --- Global indices with specified values
specifiedDegreesOfFreedom    = 0;   % --- Specified degrees of freedom values

% --- Natural boundary condition (du / dn specified) at the second end
x = globalNodes(totalNumNodes);     % --- Global coordinate of second end
value = 4 * pi * cos(4 * pi);
f(totalNumNodes) = f(totalNumNodes) + pp(x) * value; 
% f(totalNumNodes) = f(totalNumNodes) - pp(x) * val; % --- If, instead of the second end, the
                                                   %     boundary condition is enforced at the
                                                   %     first end

% --- Mixed boundary condition (du / dn = value(1) * u + value(2)) at the second end
% --- For mixed boundary condition, the value is a vector with two numbers    
% value = [1 2];
% x = globalNodes(totalNumNodes);               
% K(totalNumNodes, totalNumNodes) = K(totalNumNodes, totalNumNodes) - pp(x) * value(1);  
% f(totalNumNodes) = f(totalNumNodes) + pp(x) * value(2);                
% --- If first end, instead
% K(totalNumNodes, totalNumNodes) = K(totalNumNodes, totalNumNodes) + pp(x) * value(1);  
% f(totalNumNodes) = f(totalNumNodes) - pp(x) * value(2);                

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SOLVE THE LINEAR SYSTEM %
%%%%%%%%%%%%%%%%%%%%%%%%%%%
d = SolveSystem(K, f, globalIndicesSpecifiedValues, specifiedDegreesOfFreedom); 
  
%%%%%%%%%%%%%%%%%%%
% POST PROCESSING %
%%%%%%%%%%%%%%%%%%%
u_plot = [];

numPlotPointsPerElement  = 10;                                  % --- Number of plot points per element

plotPoints = linspace(-1, 1, numPlotPointsPerElement);

numberGaussPoints   = 2;                                        % --- Number of Gauss points

uniformNorm         = 0.0;
l2DifferenceNorm    = 0.0;                   
l2Norm              = 0.0;           

[gaussPoints, gaussWeights] = gaussianQuadrature(numberGaussPoints);         % --- Return Gauss quadrature points and weights                      

for e = 1 : numElements     % --- Loop over the elements
    
    globalNodeNumbers = connectivityMatrix(:, e);               % --- Global node numbers corresponding to the current element 
    xe = globalNodes(globalNodeNumbers)';                       % --- Global coordinates of the element nodes as a column vector

    de = d(globalNodeNumbers);                                  % --- Restores the node values
      
    % --- Loop over all the points for plotting
    for kk = 1 : length(plotPoints)    

        Ne = LinearBasisFunctions1D(plotPoints(kk));            % --- Return the 1D linear basis functions along with their derivatives with respect to local coordinates xi at 

        x   = Ne * xe;                                          % --- Global coordinate (here x) of the current integration point
        uh  = Ne * de;                                          % --- Unknown value at plot points
        u_plot  = [u_plot; x uh];

    end
    
    % --- Loop over all the Gauss points for error analysis
    for kk = 1 : length(gaussPoints)         

        [Ne, dNe] = LinearBasisFunctions1D(gaussPoints(kk));    % --- Return the 1D linear basis functions along with their derivatives with respect to local coordinates xi at 
                                                                %     Gauss points
           
        x   = Ne * xe;                                          % --- Global coordinate (here x) of the current integration point
                                      
        Jacobian     = dNe * xe;                                % --- Jacobian dx / dxi
            
        JacxW = Jacobian * gaussWeights(kk);                    % --- Calculate the integration weight
            
        u_exact = exact(x);                                     % --- Analytical solution

        uh = Ne * de;                                           % --- Finite element solution at the Gauss points

        % --- Uniform norm
        error = abs(uh - u_exact);
        if (error > uniformNorm)
            uniformNorm = error;     
        end
             
        l2DifferenceNorm = l2DifferenceNorm + (u_exact - uh)^2 * JacxW;
        l2Norm = l2Norm + u_exact^2*JacxW;
    end
end

% --- Exact analytical solution at plot points
x_plot = linspace (globalNodes(1), globalNodes(totalNumNodes), 100); 
u_exact = exact(x_plot);                       

fprintf(1,'\tUniform norm: %e\n', uniformNorm);
fprintf(1,'\tPercentage root mean square error: %e\n', 100 * l2DifferenceNorm);

%%%%%%%%%
% GRAPH %
%%%%%%%%%
figure(1)
plot(u_plot(:,1), u_plot(:,2), '-.r');
hold on;
plot(x_plot, u_exact, '-k'); legend('FEM','Exact Solution');
ylabel('u');  xlabel('x'); 
title('u: FEM versus analytical solution');



