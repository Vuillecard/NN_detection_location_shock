function [ data ] = data_gen_all_neighborhood_orthogonal(alph ,fun,Net,Mesh,Ainv ,b, remove_duplon )
% Generate data for a given mesh all neighborhood take information of all
% the three neighborhood triangle. The data is generate to work is the
% location detection task.
% remove_duplon (bool): true to remove similar data


    % Compute vertexes
    VA = [Mesh.x(1,:); Mesh.y(1,:)];
    VB = [Mesh.x(Mesh.N+1,:); Mesh.y(Mesh.N+1,:)];
    VC = [Mesh.x(end,:); Mesh.y(end,:)];

    % Evaluate solution
    Q = feval(fun, Mesh.x, Mesh.y);
    
    % Boundary conditions
    QG = ApplyBCScalar2D(Q,0,Mesh);
    
    % Evaluate the network
    nconst_ind = FindNonConstCells2D(Q);
    ind = NN_Indicator2D(Q,QG,Mesh,Net,nconst_ind);
    
    % Get Neighbours
    E1 = Mesh.EToE(:,1)'; E2 = Mesh.EToE(:,2)'; E3 = Mesh.EToE(:,3)';   

    % Get modal coefficients of patch
    Qm1 = Q(:,E1); Qm2 = Q(:,E2); Qm3 = Q(:,E3);
    
    Q_all = [ Q ; Qm1 ; Qm2 ; Qm3 ];
    
    % Set angle
    angl_phy(ind) = alph;
    angl_phy(setdiff(1:Mesh.K,ind)) = NaN;

    % Map from physical to reference
    angl_ref = atan2(Ainv(3,:).*cos(angl_phy)+Ainv(4,:).*sin(angl_phy),Ainv(1,:).*cos(angl_phy)+ Ainv(2,:).*sin(angl_phy));
   
    
    % find the location of the discontinuity
    point_intersection_ref = ones(2,length(ind)) ;
    vec_min_dist_ref = ones(2,length(ind)) ;
    j = 1;
    for i = ind 
        Y = 0 ;
        
        if fun(VA(1,i),VA(2,i)) ~= fun(VB(1,i),VB(2,i)) 
            Y = VB(:,i);
        elseif fun(VA(1,i),VA(2,i)) ~= fun(VC(1,i),VC(2,i))
            Y = VC(:,i);
        else
            disp(" Discontinuity not found ")
        end
        % try to solve a line intersection :
        % where line are like :
        % p1 = p2 + lambda*v

        alph_2 = atan2(Y(2)-VA(2,i),Y(1)-VA(1,i));
        T11 = cos(alph);
        T12 = -cos(alph_2);
        T21 = sin(alph) ;
        T22 = -sin(alph_2);
        determinant_T = T11.*T22 - T12.*T21;
        %Tinv11 = T22./determinant_T;
        %Tinv12 = -T12./determinant_T;
        Tinv21 = -T21./determinant_T;
        Tinv22 = T11./determinant_T;
        
        dir_2 = [ cos(alph_2) ; sin(alph_2) ] ;
        dist_2 = Tinv21*VA(1,i)+ Tinv22*VA(2,i) ;
        point_intersection = VA(:,i)+dist_2*dir_2 ;
        %dir_1 = [ cos(alph) ; sin(alph) ] ;
        %dist_1 = Tinv11*VA(1,i)+ Tinv12*VA(2,i) ;
        %point_intersection(:,j) = dist_1*dir_1 ;
        point_intersection_ref(:,j) = transformation_phy_to_ref(point_intersection,Ainv(:,i),b(:,i)) ;
        
        VA_ref = transformation_phy_to_ref(VA(:,i) ,Ainv(:,i),b(:,i)) ;
        
        angle_ref =  angl_ref(i) ;
        
        % Compute the distance from where the distance is minimal between
        % VA_ref and the discontinuity. 
        
        dir = [ cos(angle_ref) ; sin(angle_ref) ] ;
        dist_min = -point_intersection_ref(1,j)*cos(angle_ref) - point_intersection_ref(2,j)*sin(angle_ref) + VA_ref(1)*cos(angle_ref) + VA_ref(2)*sin(angle_ref) ;
        point_min_dist_ = point_intersection_ref(:,j)  + dist_min*dir ;
        vec_min_dist_ref(:,j) = [point_min_dist_(1) - VA_ref(1) ; point_min_dist_(2) - VA_ref(2)  ] ;
        % [point_min_dist_(1) - VA_ref(1) ; point_min_dist_(2) - VA_ref(2)  ] ;
        j =j +1;
    end 
    
   
    % Export the training set 
    data = [ind' ,point_intersection_ref' , Q_all(:,ind)'  , vec_min_dist_ref', angl_ref(ind)' ];
    
    % Remonving similar observation
    if remove_duplon
        [numrow,ncol] = size(data);
        i=1 ;
        count=0;
        while i < numrow
           liste_ind = [];
           for j = [i+1:numrow]
               if sum(abs(data(i,2:end)-data(j,2:end)))<1e-4
                   liste_ind = [liste_ind,j];
                   count = count +1;
               end
           end
           data(liste_ind,:)=[];
           [numrow,ncol] = size(data);
           i=i+1;
        end
    end
end 