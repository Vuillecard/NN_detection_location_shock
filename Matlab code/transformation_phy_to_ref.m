function [ x_ref] = transformation_phy_to_ref(x ,Ainv,b)
% transformation from pysical space to the reference space
    x_ref_1 = Ainv(1)*(x(1)-b(1)) + Ainv(2)*(x(2)-b(2)) ;
    x_ref_2 = Ainv(3)*(x(1)-b(1)) + Ainv(4)*(x(2)-b(2)) ;
    x_ref = [x_ref_1 ; x_ref_2 ] ;
    
end