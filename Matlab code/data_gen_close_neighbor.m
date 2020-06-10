function [ data ] = data_gen_close_neighbor(alph ,fun,Net,Mesh,Ainv11,Ainv12,Ainv21,Ainv22,remove_duplon,circle)
% Generate data for a given mesh all neighborhood take edges information of all
% the three neighborhood triangle
% remove_duplon (bool): true to remove similar data
% circle (bool): true if the discontinuity is a circle

    % Evaluate solution
    Q = feval(fun, Mesh.x, Mesh.y);

    % Boundary conditions
    QG = ApplyBCScalar2D(Q,0,Mesh);

    % Evaluate the network
    nconst_ind = FindNonConstCells2D(Q);
    ind = NN_Indicator2D(Q,QG,Mesh,Net,nconst_ind);
    
    % Value on boundary from neighboring elements
    Mesh.vmapM = reshape(Mesh.vmapM, Mesh.Nfp*Mesh.Nfaces, Mesh.K); 
    Mesh.vmapP = reshape(Mesh.vmapP, Mesh.Nfp*Mesh.Nfaces, Mesh.K);
    QP = Q(Mesh.vmapP);
    
    Q_all = [ Q ; QP ];
    
    % Set angle
    if circle 
        angl_ref(1:Mesh.K)= NaN ;
    else
        angl_phy(ind) = alph;
        angl_phy(setdiff(1:Mesh.K,ind)) = NaN;

        % Map from physical to reference
        angl_ref = atan2(Ainv21.*cos(angl_phy)+Ainv22.*sin(angl_phy),Ainv11.*cos(angl_phy)+Ainv12.*sin(angl_phy));
    end
    % Export the training set 
    data = [ind',Q_all(:,ind)' , angl_ref(ind)' ];
    
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