function [ x_phy] = transformation_ref_to_phy(x_ref ,A,b)
% transformation from reference space to the physical space
    x_phy_1 = A(1)*x_ref(1) + A(2)*x_ref(2) + b(1);
    x_phy_2 = A(3)*x_ref(1) + A(4)*x_ref(2) + b(2);
    x_phy = [x_phy_1 ; x_phy_2 ] ;
    
end