function AssembleGlobalMatrix(e, Ke, fe);

global numNodesPerElement numDOFPerNode connectivityMatrix f I J X ntriplets

% --- The code is ready to be extended from 2D to 3D by acting on
%     numDOFPerNode. The degrees of freedom are numbered sequentially for x, y ,z directions 
%     starting from node 1. Thus, for the ith node the DOF are (ii-1)*numDOFPerNode + a, 
%     a = 1,..,numDOFPerNode.

for ii = 1 : numNodesPerElement             % --- Loop over all the nodes of the e-th element
    
    for a = 1 : numDOFPerNode               % --- Loop over all the degrees of freedom of the e-th element
        
        localIndexii = (ii - 1) * numDOFPerNode + a;                      

        globalIndexii = numDOFPerNode * (connectivityMatrix(ii, e) - 1) + a;   % --- Global index of the ii-th local node of the e-th element
       
        if (fe(localIndexii)~=0)
            f(globalIndexii) = f(globalIndexii) + fe(localIndexii);    % --- Assemble load
        end
        
        for jj = 1 : numNodesPerElement                              % --- Loop over all the nodes of the e-th element
            
            for b = 1 : numDOFPerNode                                % --- Loop over all the degrees of freedom of the e-th element
                
                localIndexjj = (jj - 1) * numDOFPerNode + b;              
                
                globalIndexjj = numDOFPerNode * (connectivityMatrix(jj, e) - 1) + b;   % --- Global index of the ii-th local node of the e-th element

                % --- If the element (ii, jj) of the stiffness matrix of the e-th
                %     element is different from zero, then add a triplet
                if (Ke(localIndexii, localIndexjj) ~= 0)            
                    
                    ntriplets = ntriplets + 1;
                                        
                    % --- Manually increase the size of the I, J, X arrays if needed, namely, it ntriplets > len. 
                    %     The below approach speeds up this process. All the values of these arrays 
                    %     between len and 2*len are set to zero and the length of these arrays is increased to 2*len    
                    len = length(X);          
                    if (ntriplets > len)  
                        I(2 * len) = 0;   
                        J(2 * len) = 0;    
                        X(2 * len) = 0;     
                    end

                    I(ntriplets) = globalIndexii;
                    J(ntriplets) = globalIndexjj;
                    X(ntriplets) = Ke(localIndexii, localIndexjj);

                end
            end
        end
    end
end



