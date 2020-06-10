function [ data] = data_gen(alph ,fun,Net,Mesh,Ainv11,Ainv12,Ainv21,Ainv22,remove_duplon,circle)
% Generate data for a given mesh
% remove_duplon (bool): true to remove similar data
% circle (bool): true if the discontinuity is a circle

    % Evaluate solution
    Q = feval(fun, Mesh.x, Mesh.y);

    % Boundary conditions
    QG = ApplyBCScalar2D(Q,0,Mesh);

    % Evaluate the network
    nconst_ind = FindNonConstCells2D(Q);
    ind = NN_Indicator2D(Q,QG,Mesh,Net,nconst_ind);
    
    if circle 
        % set the angle to zero since it's hard to compute the true angle 
        angl_ref(1:Mesh.K) = NaN;
    else
        % Set angle
        angl_phy(ind) = alph;
        angl_phy(setdiff(1:Mesh.K,ind)) = NaN;

        % Map from physical to reference
        angl_ref = atan2(Ainv21.*cos(angl_phy)+Ainv22.*sin(angl_phy),Ainv11.*cos(angl_phy)+Ainv12.*sin(angl_phy));
    end
    % Export the training set 
    % [data_tmp, ia , ic]= unique([Q(:,ind)' , angl_ref(ind)' ],'rows');
    % data = [ ind(ia)' , data_tmp ] ; 
    
    data_tmp = [Q(:,ind)' , angl_ref(ind)' ];
    data = [ ind' , data_tmp ] ;
    
    % Remonving similar observation
    if remove_duplon
        [numrow,ncol] = size(data);
        i=1 ;
        count=0;
        while i < numrow
           liste_ind = [];
           % search for similar observation in the data 
           for j = [i+1:numrow]
               if sum(abs(data(i,2:end)-data(j,2:end)))<1e-4
                   liste_ind = [liste_ind,j];
                   count = count +1;
               end
           end
           % remove similar observation from data
           data(liste_ind,:)=[];
           [numrow,ncol] = size(data);
           i=i+1;
        end
    end
    
end 