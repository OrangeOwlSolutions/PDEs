%  --- Solves a two dimensional plane stree problem for a bending of a cantilever beam loaded at the end.
%      ^
%      |
%      |                                             
%     y|                                              /
%      -----------------------------------------------/       
%      |                                              /
%      |                                              /
%     x------------->        2c                       /          
%      |                                              /
%      |                     L                        /
%      -----------------------------------------------/
%                                                     /
%  with the following boundary conditions
%  left: force P
%  right : fixed end
%  other sides: traction free

clc;
clear all;
close all; 

global dim totalNumNodes numElements numNodesPerElement numDOFPerNode numEquations boundaryConditionsNodes boundaryConditionsValues D E nu l c P I J X ntriplets K f d ...   
       globalNodes connectivityMatrix IsBoundaryCondition BoundaryNodes

xl = 0.0;                               % --- Left end of the boundary along x
xr = 10.0;                              % --- Right end of the boundary along x
yb = -0.5;                              % --- Bottom end of the range along y
yt = 0.5;                               % --- Top end of the range along y
    
c  = 0.5;                               % --- beam width
l  = 10;                                % --- beam length

E  = 1.0e7;                             % --- Young's modulus
nu = 0.3;                               % --- Possion's ratio

P = 100.0;                              % --- Point load on the right boundary

% --- Constitutive matrix
% --- Plane stress
D = E/(1-nu^2)*[1 ,  nu,    0;
                nu,   1,    0;
                0 ,   0,  (1-nu)/2];
% --- Plane strain
% D = E/((1+nu)*(1-2*nu))*[1-nu,    nu,  0;
%                            nu,  1-nu,  0;
%                            0 ,     0,(1-2*nu)/2];

% --- numElementsx and numElementsy are the subdivisions in the x and y directions
%     (rectangular boxes) and not the number of elements
numElementsx = 100;                     % --- Number of elements along x
numElementsy = 10;                      % --- Number of elements along y
    
dim = 2;                                % --- Number of spatial dimensions (2 (default) or 3)

numDOFPerNode = 2;                      % --- Degrees-of-freedom per node (you need to modify
                                        %     to 2 or 3 for 2D and 3D deformation problems)

setNodesConnectivity2D(xl, xr, yb, yt, numElementsx, numElementsy);     % --- Generate the grid

numEquations  = totalNumNodes * numDOFPerNode;      % --- Total number of equations 

f = zeros(numEquations, 1);                         % --- Global force vector
d = zeros(numEquations, 1);                         % --- Global solution vector

% --- Triplet for assembling the sparse matrix
I = zeros(numNodesPerElement * numEquations,1);     % --- Row indices of non-zero entries
J = zeros(numNodesPerElement * numEquations,1);     % --- Column indices of non-zero entries
X = zeros(numNodesPerElement * numEquations,1);     % --- Non-zero entries matrix

ntriplets = 0;

localDOF = numNodesPerElement * numDOFPerNode;              % --- Degrees of freedom in each element 
                    
[gaussPoints, gaussWeights] = gaussianQuadrature;           % --- Return Gauss quadrature points and weights 

[NQ, dNQ] = LinearBasisFunctions2D(gaussPoints);            % --- Return the 1D linear basis functions along with their derivatives with respect to local coordinates 
                                                            %     (xi, etai) at Gauss points
                       
%%%%%%%%%%%%
% FEM LOOP %
%%%%%%%%%%%%
for e = 1 : numElements     % --- Loop over the elements

    Ke = zeros(localDOF);                               % --- Element stiffness matrix
    fe = zeros(localDOF, 1);                            % --- Element force (load) vector 

    globalNodeNumbers = connectivityMatrix(:, e);       % --- Global node numbers corresponding to the current element 
    xe = globalNodes(:, globalNodeNumbers)';            % --- Global coordinates of the element nodes as a column vector

    area = [-1 1 0; -1 0 1] * xe;                       % --- Its determinant is twice the area 
                                                        %     of the triangle.
                                                        %     It is the jacobian of the triangle.
    xe = [xe; xe(3, :)];                                % --- Append the last node, so making a virtually
                                                        %     4-noded element. In this way, one can use the
                                                        %     basis functions of quadrilateral elements
                                                        
    % --- Calculate the element integral
    for kk = 1 : length(gaussPoints)   % --- Loop over all the Gauss points

        Ne  = NQ(kk,:);
        dNe = [dNQ(kk,:); dNQ(kk + numNodesPerElement, :)];
        
        x   = Ne * xe;                 % --- Global coordinates of the current integration point

        Jacobian = dNe * xe;           % --- Jacobian

        Jacxw = det(Jacobian) * gaussWeights(kk);     % --- Calculate the integration weight

        dx = Jacobian \ dNe;           % --- Calculate the derivatives of the basis
                                       %     function with respect to x and y direction in
                                       %     physical coordinates
            
        Ne = [Ne(1: 2), Ne(3) + Ne(4)];% --- Add the contribution of 4th node to 
                                       %     the 3rd node (use quadrilateral basis functions
                                       %     to get the triangular ones)
                                       
        dNe = [dx(1, 1 : 2), dx(1, 3) + dx(1, 4);
        
        dx(2, 1 : 2), dx(2, 3) + dx(2, 4)];
                   
        Jacxw = det(area) / 2 * gaussWeights(kk);   % --- Jacobian times integration weights 
                                                    %     of the triangle

        Ne = [Ne(1),   0    ,    Ne(2),   0    ,   Ne(3),    0;
              0    ,   Ne(1),    0    ,   Ne(2),   0    ,    Ne(3)];
    
        B  = [dNe(1, 1), 0        , dNe(1, 2), 0        , dNe(1, 3), 0;
              0        , dNe(2, 1), 0        , dNe(2, 2),         0, dNe(2, 3);
              dNe(2, 1), dNe(1, 1), dNe(2, 2), dNe(1, 2), dNe(2, 3), dNe(1, 3)];

        Ke = Ke + B' * D * B * Jacxw;

        [fx, fy] = ff(x);           % --- Extract the body force

        fe = fe + Ne' * [fx; fy] * Jacxw;

    end

    AssembleGlobalMatrix(e, Ke, fe);               % --- Assemble global matrix

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% APPLY BOUNDARY CONDITIONS %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
IsBoundaryCondition = [0 1 0 1];

boundaryConditionsNodes  = [];  
boundaryConditionsValues = [];       

% --- Side 2
for nodeID = 1 : length(BoundaryNodes(2).globalNodes)
    dof = [numDOFPerNode*(BoundaryNodes(2).globalNodes(nodeID) - 1) + 1 ...
           numDOFPerNode*(BoundaryNodes(2).globalNodes(nodeID) - 1) + 2];
    val = [0; 0];                                    
    boundaryConditionsNodes = [boundaryConditionsNodes, dof];
    boundaryConditionsValues = [boundaryConditionsValues; val];
end
ApplyNaturalBC(2);           % apply natural boundary conditions if

% --- Side 4
ApplyNaturalBC(4);           % apply natural boundary conditions if

% --- Since the corner nodes are at two boundary sides, we need to find the
%     unique numbers of boundaryConditionsNodes to avoid applying the same EBC twice
[boundaryConditionsNodes, m, NQ] = unique(boundaryConditionsNodes);
boundaryConditionsValues = boundaryConditionsValues(m);

% --- Constructs the global stiffness matrix in sparse format using I, J, X
K = sparse(I(1 : ntriplets), J(1 : ntriplets), X(1 : ntriplets), numEquations, numEquations);

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SOLVE THE LINEAR SYSTEM %
%%%%%%%%%%%%%%%%%%%%%%%%%%%
d = SolveSystem(K, f, boundaryConditionsNodes, boundaryConditionsValues); % Solution of linear systems of equations

u = d(1 : 2 : numEquations - 1);
v = d(2 : 2 : numEquations);

%%%%%%%%%%%%%%%%%%%%%%
% PLOT DEFORMED MESH %
%%%%%%%%%%%%%%%%%%%%%%
displacements = zeros(totalNumNodes, dim);

for e = 1 : totalNumNodes
    for j = 1 : dim
        displacements(e, j) = d(dim * (e - 1) + j);
    end
end

a = zeros(dim);     % --- Find the scale factor for deformation shape
for e = 1 : dim
    a(e) = max(globalNodes(e, :)) - min(globalNodes(e, :));
end
aa = max(a);
b = zeros(dim);
scale = 0;

for e = 1 : dim
    b(e) = max(displacements(:, e)) - min(displacements(:, e));
end
bb = max(b);
scale = 0.1 * aa / bb;

figure;
hold on;

for e = 1 : numElements

    globalNodeNumbers = connectivityMatrix(:, e)';
    XX = [globalNodes(1, globalNodeNumbers), globalNodes(1, globalNodeNumbers(1))];
    YY = [globalNodes(2, globalNodeNumbers), globalNodes(2, globalNodeNumbers(1))];
    plot(XX,YY);hold on;

    displacedXX = globalNodes(1, globalNodeNumbers) + scale * displacements(globalNodeNumbers,1)';
    displacedYY = globalNodes(2, globalNodeNumbers) + scale * displacements(globalNodeNumbers,2)';
    displacedXX = [displacedXX, displacedXX(1)];
    displacedYY = [displacedYY, displacedYY(1)];
    plot (displacedXX,displacedYY,'LineStyle','--');

end

legend('Initial shape', 'Deformed shape'); axis image;
title('Deformed bar'); xlabel('x'); ylabel('y');

%%%%%%%%%%%%%%%%%%
% ERROR ANALYSIS %
%%%%%%%%%%%%%%%%%%
maxerror = zeros(2, 1);
anal = zeros(totalNumNodes, 2);
    
for i = 1 : totalNumNodes
    
    x = globalNodes(:, i);
        
    disp = exact(x);
                 
    error1 = abs(disp(1)-u(i));
    error2 = abs(disp(2)-v(i));
        
    if (error1 > maxerror(1))
        maxerror(1) = error1;
    end
        
    if (error2 > maxerror(2))
        maxerror(2) = error2;
    end
               
    anal(i, 1) = disp(1);
    anal(i, 2) = disp(2);
                
end

fprintf(1,' The maximum error of u displacement is %e\n', maxerror(1));
fprintf(1,' The maximum error of v displacement is %e\n', maxerror(2));


