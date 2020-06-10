function [] = pred_plot_ortho(data ,pred , Mesh,A,Output,b)
% Plot the results of a given prediction, plot the orientation and the location of the
% discontinuity

% evaluate solution 
ind = data(:,1)';
j=1;
point_intersection_phy = ones(2,length(ind)) ;
for i = ind
    
    %if data(j,2)~=data(j,5)
     %   dir_1 = [1 ; 0];
    %else
    %    dir_1 = [ 0;1];
    %end
    %B = [pred(j,1)*pred(j,3) ,pred(j,2)*pred(j,3) ];
    %T11 = dir_1(1);
    %T12 = pred(j,2);
    %T21 = dir_1(2);
    %22 = -pred(j,1);
    %determinant_T = T11.*T22 - T12.*T21;
    %Tinv11 = T22./determinant_T;
    %Tinv12 = -T12./determinant_T;
    %Tinv21 = -T21./determinant_T;
    %Tinv22 = T11./determinant_T;
    %lam_1 = Tinv11*B(1)+Tinv12*B(2);
    %x_inter_ref = [-1+lam_1*dir_1(1) ; -1 + lam_1*dir_1(2)];
    %point_intersection_phy(:,j)= transformation_ref_to_phy(x_inter_ref,A(:,i),b(:,i));
    test_angle = atan2(pred(j,1),pred(j,2));
    if test_angle <= pi/4
        lam = pred(j,3)/cos(test_angle);
        x_inter_ref = [-1+lam ; -1 ]
    else
       if test_angle <= pi/2
            lam = pred(j,3)/cos((pi/2)-test_angle);
       else
           lam = pred(j,3)/cos(test_angle-(pi/2));
       end
       x_inter_ref = [-1; -1+ lam ];
    end
    
    point_intersection_phy(:,j)= transformation_ref_to_phy(x_inter_ref,A(:,i),b(:,i));
    B = [pred(j,1)*pred(j,3) ,pred(j,2)*pred(j,3) ];
    if j < 20 
        VA_ref = [-1;-1];
        VB_ref = [1;-1];
        VC_ref = [-1;1];
        
        %figure;
        %lam_ = linspace(-3 , 3 , 1000);
        
        %plot(B(1)-1 - lam_*pred(j,2) ,B(2)-1 + lam_*(pred(j,1)) )
        %hold on
        %plot(VA_ref(1), VA_ref(2), 'b.');
        %plot(VB_ref(1), VB_ref(2), 'b.');
        %plot(VC_ref(1), VC_ref(2), 'b.');
        %plot([VA_ref(1) VB_ref(1)],[VA_ref(2) VB_ref(2)],'g');
        %plot([VA_ref(1) VC_ref(1)],[VA_ref(2) VC_ref(2)],'g');
        %plot([VB_ref(1) VC_ref(1)],[VB_ref(2) VC_ref(2)],'g');
        %plot(x_inter_ref(1),x_inter_ref(2),'rx');
        %plot(B(1)-1,B(2)-1,'rx');
        
    end
    j=j+1;
    
end

pred_alpha_ref_ =  atan2(-pred(:,2),pred(:,1))';
pred_angle_phy = atan2(A(3,ind).*cos(pred_alpha_ref_)+A(4,ind).*sin(pred_alpha_ref_),A(1,ind).*cos(pred_alpha_ref_)+A(2,ind).*sin(pred_alpha_ref_));


x= linspace(-(1/sqrt(2)),(1/sqrt(2)),1000);

figure
quiver(point_intersection_phy(1,:),point_intersection_phy(2,:),cos(pred_angle_phy),sin(pred_angle_phy),0.1,'b');
xlim(Output.xran); ylim(Output.yran);
title('predicted circular shock')

figure
quiver(point_intersection_phy(1,:),point_intersection_phy(2,:),cos(pred_angle_phy),sin(pred_angle_phy),0.1,'b');
hold on
plot(x , sqrt(0.5-x.^2),'r')
plot(x , -sqrt(0.5-x.^2),'r')
xlim(Output.xran); ylim(Output.yran);
title('zoom on predicted circular shock')


end 
