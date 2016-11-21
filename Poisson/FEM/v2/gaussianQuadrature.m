function [gaussPoints, gaussWeights] = gaussianQuadrature(ngp)

% --- 1 Gauss point
if (ngp == 1)       
    gaussPoints     = 0;
    gaussWeights    = 2;

% --- 2 Gauss points
elseif (ngp == 2)   
    gaussPoints(1)  = - 0.577350269189625764509148780502;
    gaussPoints(2)  =   0.577350269189625764509148780502;

    gaussWeights(1) = 1.0;
    gaussWeights(2) = 1.0;

% --- 3 Gauss points
elseif (ngp ==3)    

    gaussPoints(1)  = - 0.774596669241483377035853079956;
    gaussPoints(2)  =   0.0;
    gaussPoints(3)  =   0.774596669241483377035853079956;

    gaussWeights(1) = 5.0 / 9.0;
    gaussWeights(2) = 8.0 / 9.0;
    gaussWeights(3) = 5.0 / 9.0;
    
% --- 4 Gauss points
elseif (ngp == 4) 

    gaussPoints(1)  = - 0.861136311594052575223946488893;
    gaussPoints(2)  = - 0.339981043584856264802665759103;
    gaussPoints(3)  =   0.339981043584856264802665759103;
    gaussPoints(4)  =   0.861136311594052575223946488893;

    gaussWeights(1) = 0.347854845137453857373063949222;
    gaussWeights(2) = 0.652145154862546142626936050778;
    gaussWeights(3) = 0.652145154862546142626936050778;
    gaussWeights(4) = 0.347854845137453857373063949222;

% --- 5 Gauss points
elseif (ngp == 5) 

    gaussPoints(1)  = - 0.906179845938663992797626878299;
    gaussPoints(2)  = - 0.538469310105683091036314420700;
    gaussPoints(3)  =   0.0;
    gaussPoints(4)  =   0.538469310105683091036314420700;
    gaussPoints(5)  =   0.906179845938663992797626878299;

    gaussWeights(1) = 0.236926885056189087514264040720;
    gaussWeights(2) = 0.478628670499366468041291514836;
    gaussWeights(3) = 0.568888888888888888888888888889;
    gaussWeights(4) = 0.478628670499366468041291514836;
    gaussWeights(5) = 0.236926885056189087514264040720;

else

    fprintf ( 1, '\n');
    fprintf ( 1, 'Gaussian quadrature - Fatal error!\n' );
    fprintf ( 1, '  Illegal number of Gauss points = %d\n', ngp );
    fprintf ( 1, '  Legal values are 1 to 5.\n' );

end



  
