function setNodesConnectivity2D(xl, xr, yb, yt, numElementsx, numElementsy);

global dim totalNumNodes numElements numNodesPerElement globalNodes connectivityMatrix BoundaryNodes BoundaryElems  

numElements = 2 * numElementsx * numElementsy;              % --- Total number of elements
numNodesPerElement = 3;                                     % --- Number of nodes per e

totalNumNodes = (numElementsx + 1) * (numElementsy + 1);    % --- Total number of nodes
                              
globalNodes = zeros(dim, totalNumNodes);                          % --- Node coordinates array
connectivityMatrix = zeros(numNodesPerElement, numElements);% --- Element connectivity matrix

% --- Node coordinates
x = linspace(xl, xr, numElementsx + 1);           
y = linspace(yb, yt, numElementsy + 1);          

for j = 0 : numElementsy                      % --- Loop over all nodes
    for i = 0 : numElementsx

        nodeID = (numElementsx + 1) * j + i + 1;

        globalNodes(1, nodeID) = x(i + 1);       
        globalNodes(2, nodeID) = y(j + 1);       
    end
end

% --- For each component, BoundaryNodes is a vector containing all global node numbers on the
%     corresponding boundary
% --- First boundary (bottom)
BoundaryNodes(1).globalNodes = [1 : numElementsx + 1];  

% --- Second boundary (right)
BoundaryNodes(2).globalNodes = [1 : (numElementsy + 1)] * (numElementsx + 1);

% --- Third boundary (top)
BoundaryNodes(3).globalNodes = (numElementsx + 1) * numElementsy  + [1 : numElementsx + 1];

% --- Fourth boundary (left)
BoundaryNodes(4).globalNodes = (numElementsx + 1) * [0 : numElementsy] + 1;

% --- Loop over all elements to set the node connectivity matrix
e = 0;
for j = 1 : numElementsy
    for i = 1 : numElementsx

        sw = i     + (j - 1) * (numElementsx + 1);
        se = i + 1 + (j - 1) * (numElementsx + 1);
        nw = i     +  j      * (numElementsx + 1);
        ne = i + 1 +  j      * (numElementsx + 1);
       
        % --- Recall that each rectangle is split into two triangles
        e = e + 1;

        connectivityMatrix(1, e) = sw;
        connectivityMatrix(2, e) = se;
        connectivityMatrix(3, e) = nw;

        e = e + 1;

        connectivityMatrix(1, e) = ne;
        connectivityMatrix(2, e) = nw;
        connectivityMatrix(3, e) = se;

    end
end

% --- Set up the boundary and surface indicators for elems - find elements that have sides 
%     on the boundaries

% --- For each component, BoundaryElems is a vector containing all elements having sides on the
%     corresponding boundary
% It also contains a vector of surface indicator which indicates the
% surface of the local finite elemnts according to the order in the
% BoundaryElem(i).connectivityMatrix vector.
%
%       BoundaryElems(i). SurfaceInidcator = [].

% --- First boundary (bottom)
BoundaryElems(1).Elems = [1 : 2 : 2 * numElementsx];
BoundaryElems(1).SurfaceIndicator = -2 * ones(numElementsx, 1);

% --- Second boundary (right)
BoundaryElems(2).Elems = [1 : numElementsy] * 2 * numElementsx;
BoundaryElems(2).SurfaceIndicator = -1 * ones(numElementsy, 1);

% --- Third boundary (top)
BoundaryElems(3).Elems =  2 * numElementsx * (numElementsy - 1) + 1 + [1 : 2 : 2 * numElementsx];
BoundaryElems(3).SurfaceIndicator = -2 * ones(numElementsx, 1);

% --- Fourth boundary (left)
BoundaryElems(4).Elems = 2 * numElementsx * [0 : (numElementsy - 1)] + 1;
BoundaryElems(4).SurfaceIndicator = -1 * ones(numElementsy, 1);






