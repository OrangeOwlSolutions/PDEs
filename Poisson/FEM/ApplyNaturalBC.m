function  ApplyNaturalBC(sideInd);

global numNodesPerElement numDOFPerNode globalNodes connectivityMatrix BoundaryElems c P

localDOF = numNodesPerElement * numDOFPerNode;       % --- Degrees of freedom in each element 

BElems = BoundaryElems(sideInd). Elems;
SurfID = BoundaryElems(sideInd).SurfaceIndicator;

% --- Return Gauss quadrature points and weights (2 points for one dimension)
gaussPoints  = [-1 / sqrt(3), 1 / sqrt(3)];         
gaussWeights = [1, 1];            
   
for elmID = 1 : length(BoundaryElems(sideInd).Elems)
    
    Ke = zeros(localDOF);    
    fe = zeros(localDOF, 1);  

    globalNodeNumbers = connectivityMatrix(:, BElems(elmID));   % --- Global node numbers corresponding to the current element 

    xe = globalNodes(:, globalNodeNumbers)';                 % --- Global coordinates of the element nodes as a column vector

    % --- Calculate the element integral
    for kk = 1 : length(gaussWeights)             % --- Loop over all the Gauss points

        % --- Different basis functions according to different sides
        switch (SurfID(elmID))
            case -2
                Ne = [(1 - gaussPoints(kk))/2, (1 + gaussPoints(kk))/2, 0];
                dNe = [-1/2, 1/2, 0];
            case +1
                Ne = [0, (1 - gaussPoints(kk))/2, (1 + gaussPoints(kk))/2];
                dNe = [0, -1/2, 1/2];
            case -1
                Ne = [(1 + gaussPoints(kk))/2, 0, (1 - gaussPoints(kk))/2];
                dNe = [1/2, 0, -1/2];
        end

        x = Ne * xe;            % --- Global coordinates of the current integration point

        Jacobian = dNe * xe;      % --- Jacobian   
   
        detJ = norm(Jacobian);    
   
        Jacxw = detJ * gaussWeights(kk); % --- Calculate the integration weight
   
        % --- Direct cosines
        nx = Jacobian(2) / detJ;    
        ny = -Jacobian(1) / detJ;
   
        % User provides the integrands4side in the following function
%         fe = integrands4side(sideInd, Jacxw, nx, ny, x, Ne, Ke, fe);
   
        % --- Specify the natural boundary condition (if any)
        qn = 0;                          
        qt = 0;                           

        if (sideInd == 4) 
            qn  = 0;
            qt  = -P / (2 * c);
        end

        N = [Ne(1),      0,    Ne(2),      0,     Ne(3),        0;
                 0,  Ne(1),        0,  Ne(2),         0,    Ne(3)];

        % --- Transform to x and y direction using the following formula
        %     where nx and ny are the direction cosines at the current boundary
        q = [nx -ny; ny  nx] * [qn; qt];
         
        % --- Assemble into force vector
        fe = fe +  N' * q * Jacxw;

    end
    
    AssembleGlobalMatrix(BElems(elmID), Ke, fe);      % --- Assemble global matrix
   
end
