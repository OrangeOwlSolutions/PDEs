function AssembleGlobalMatrix(e, Ke, fe);

global numNodesPerElement connectivityMatrix f I J X ntriplets

for ii = 1 : numNodesPerElement                         % --- Loop over all the nodes of the e-th element
    
    globalIndexii = connectivityMatrix(ii, e);          % --- Global index of the ii-th local node of the e-th element
       
    if (fe(ii)~=0)
        f(globalIndexii) = f(globalIndexii) + fe(ii);   % --- Assemble load
    end
        
    for jj = 1 : numNodesPerElement                     % --- Loop over all the nodes of the e-th element
        
        globalIndexjj = connectivityMatrix(jj, e);      % --- Global index of the ii-th local node of the e-th element
            
        % --- If the element (ii, jj) of the stiffness matrix of the e-th
        % element is different from zero, then add a triplet
        if (Ke(ii, jj) ~= 0)                            
            
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
            X(ntriplets) = Ke(ii, jj);
        end
    end
end



