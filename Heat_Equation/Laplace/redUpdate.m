function redUpdate(omega, NUM)

global aP aW aE aS aN b temp L2DifferenceArray                                     
	
for tidx = 1 : NUM,                                             % --- Index for the coefficients
    for tidy = 1 : NUM,                                         % --- Index for the coefficients
        
        tidx1 = tidx + 1;                                       % --- Index for the "red" image
        tidy1 = tidy + 1;                                       % --- Index for the "red" image
        
        if (mod(tidx1 + tidy1, 2) == 0) 
            
            temp_old = temp(tidx1, tidy1);

            res = b(tidx, tidy) ...
                + (aW(tidx, tidy) * temp(tidx1,     tidy1 - 1) ...
                +  aE(tidx, tidy) * temp(tidx1,     tidy1 + 1) ...
                +  aS(tidx, tidy) * temp(tidx1 - 1, tidy1    ) ...
                +  aN(tidx, tidy) * temp(tidx1 + 1, tidy1    ));
            
            temp_new = temp_old * (1 - omega) + omega * (res / aP(tidx, tidy));

            temp(tidx1, tidy1) = temp_new;
		
            res = temp_new - temp_old;
            L2DifferenceArray(tidx1, tidy1) = res * res;
        end
        
		% --- If we are not on a "red" pixel, then exit.

    end
end
