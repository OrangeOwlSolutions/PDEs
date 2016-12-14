function d = SolveSystem(K, f, globalIndicesSpecifiedValues, specifiedDegreesOfFreedom)

dof = length(f);                    % --- Extract the total number of degreess-of-freedom                   

% --- Separate the degrees of freedom to the boundary condition points
%     C = setdiff(A,B) for vectors A and B, returns the values in A that are not in B with no 
%     repetitions. C will be sorted.
df = setdiff(1 : dof, globalIndicesSpecifiedValues);        
Kf = K(df, df);                    
Rf = f(df) - K(df, globalIndicesSpecifiedValues) * specifiedDegreesOfFreedom;  

% --- Solve the linear system of equations using Gauss elimination. Other (possibly more efficient)
%     solution schemes are possible, taking also advantage of the sparse matrix data structure.
dfVals = Kf \ Rf;

d = zeros(dof, 1);                  % --- Restore the solution vector
d(globalIndicesSpecifiedValues) = specifiedDegreesOfFreedom;
d(df)   = dfVals;


