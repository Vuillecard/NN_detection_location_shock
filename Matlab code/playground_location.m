 cd ./deepray-dgann
mypath
cd ..

CleanUp2D;

close all; clear all; clc;

% boundary conditions
%BC_cond = {100001,'P'; 100002,'P'; 100003,'P'; 100004,'P'};
BC_cond = {100001,'D'; 100002,'D'; 100003,'D'; 100004,'D'};
%BC_cond = {100001,'D'};
%solution to be considered

alph=120*pi/180;
InitialCond = @(x,y) 1*(y<tan(alph)*(x))+0*(y>=tan(alph)*x);

%polynomial order
N = 3;

%ranges to visualize solution
xran = [-1,1]; yran = [-1,1];

% Mesh file
% generate using >> /Applications/Gmsh.app/Contents/MacOS/gmsh <geo_filename>.geo -2 -format msh2 -o <mesh_filename>.msh 
msh_file = 'mesh_square.msh';

%parameters to be set etc (not important here)
model = 'Advection'; AdvectionVelocity = [1,1]; test_name = 'Mytest'; 
FinalTime = 1; CFL = 0.6; tstamps = 2; 
Indicator = 'NN'; TVBM = 10; TVBnu = 1.5; Filter_const = true; nn_model = 'MLP_v1'; Limiter = 'BJES';
plot_iter = 50; show_plot  = true; clines = linspace(-0.98,0.98,30); save_soln =true;

% Create BC_flag
CreateBC_Flags2D;

% Check parameters
ScalarCheckParam2D;

% Display paramaters
ScalarStartDisp2D;

% Initialize solver and construct grid and metric
[Mesh.VX,Mesh.VY,Mesh.K,Mesh.Nv,Mesh.EToV,Mesh.BFaces,Mesh.PerBToB_map,Mesh.PerBFToF_map] ...
                                          = read_gmsh_file(Mesh.msh_file);

% Generate necessary data structures
StartUp2D;

% Get essential BC_flags
Mesh.BC_ess_flags = BuildBCKeys2D(Mesh.BC_flags,Mesh.BC_ENUM.Periodic);

BuildBCMaps2D;
    
% Find relative path
REL_PATH = './deepray-dgann/';

% Extract MLP weights, biases and other parameters
Net = read_mlp_param2D(Limit.nn_model,REL_PATH);  

% Evaluate solution
Q = feval(Problem.InitialCond, Mesh.x, Mesh.y);

% Boundary conditions
QG = ApplyBCScalar2D(Q,0,Mesh);

% Evaluate the network
nconst_ind = FindNonConstCells2D(Q);
ind = NN_Indicator2D(Q,QG,Mesh,Net,nconst_ind);

% Plot Mesh and troubled cells
xavg = Mesh.AVG2D*Mesh.x;
yavg = Mesh.AVG2D*Mesh.y;
figure;
plot(xavg(ind), yavg(ind), 'rx'); 
hold on;
plot(xavg(setdiff(1:length(xavg),ind)), yavg(setdiff(1:length(yavg),ind)), 'b.'); 
xlim(Output.xran); ylim(Output.yran);

% Compute vertexes
VA = [Mesh.x(1,:); Mesh.y(1,:)];
VB = [Mesh.x(Mesh.N+1,:); Mesh.y(Mesh.N+1,:)];
VC = [Mesh.x(end,:); Mesh.y(end,:)];


% Get the affine transformation x = A.x_hat + b
% Entries of the matrix representing the geometric mapping
A11 = 1/2*(VB(1,:)-VA(1,:));
A12 = 1/2*(VC(1,:)-VA(1,:));
A21 = 1/2*(VB(2,:)-VA(2,:));
A22 = 1/2*(VC(2,:)-VA(2,:));

b1 = 1/2*(VB(1,:)+ VC(1,:));
b2 = 1/2*(VB(2,:)+VC(2,:));

b = [b1 ; b2] ; 

% Repeat with inverse
determinant = A11.*A22 - A12.*A21;
Ainv11 = A22./determinant;
Ainv12 = -A12./determinant;
Ainv21 = -A21./determinant;
Ainv22 = A11./determinant;

Ainv = [ Ainv11 ;Ainv12  ;Ainv21 ;Ainv22 ];
% Set angle
angl_phy(ind) = alph;
angl_phy(setdiff(1:Mesh.K,ind)) = NaN;

% Map from physical to reference
angl_ref = atan2(Ainv21.*cos(angl_phy)+Ainv22.*sin(angl_phy),Ainv11.*cos(angl_phy)+Ainv12.*sin(angl_phy));

% Map from reference to physical
angle_phy_check = atan2(A21.*cos(angl_ref)+A22.*sin(angl_ref),A11.*cos(angl_ref)+A12.*sin(angl_ref));


% find the location of the discontinuity
point_intersection = ones(2,length(ind)) ; 
point_min_dist = ones(2,length(ind));
j = 1;
for i = ind 
    Y = 0 ;
    %disp(InitialCond(VA(1,i),VA(2,i)))
    %disp(InitialCond(VB(1,i),VB(2,i)) )
    %disp(InitialCond(VC(1,i),VC(2,i)))
    if InitialCond(VA(1,i),VA(2,i)) ~= InitialCond(VB(1,i),VB(2,i)) 
        Y = VB(:,i);
    elseif InitialCond(VA(1,i),VA(2,i)) ~= InitialCond(VC(1,i),VC(2,i))
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
    Tinv11 = T22./determinant_T;
    Tinv12 = -T12./determinant_T;
    Tinv21 = -T21./determinant_T;
    Tinv22 = T11./determinant_T;
    dir_2 = [ cos(alph_2) ; sin(alph_2) ] ;
    dir_1 = [ cos(alph) ; sin(alph) ] ;
    dist_2 = Tinv21*VA(1,i)+ Tinv22*VA(2,i) ;
    dist_1 = Tinv11*VA(1,i)+ Tinv12*VA(2,i) ;
    %point_intersection(:,j) = dist_1*dir_1 ;
    point_intersection(:,j) = VA(:,i)+dist_2*dir_2 ;
    dist_min = -point_intersection(1,j)*cos(alph) - point_intersection(2,j)*sin(alph) + VA(1,i)*cos(alph) + VA(2,i)*sin(alph) ;
    point_min_dist(:,j) = point_intersection(:,j)  + dist_min*dir_1 ;
    
    T11 = -sin(alph);
    T12 = -cos(alph);
    T21 = cos(alph) ;
    T22 = -sin(alph);
    determinant_T = T11.*T22 - T12.*T21;
    Tinv11 = T22./determinant_T;
    Tinv12 = -T12./determinant_T;
    Tinv21 = -T21./determinant_T;
    Tinv22 = T11./determinant_T;
    
    dir_perp = [ -sin(alph) ; cos(alph) ] ;
    dist_min_1 = Tinv11*(point_intersection(1,j)-VA(1,i))+ Tinv12*(point_intersection(2,j)-VA(2,i)) ;
    dist_min_2 = Tinv21*(point_intersection(1,j)-VA(1,i))+ Tinv22*(point_intersection(2,j)-VA(2,i)) ;
    
    
    j =j +1;
       
end

%plot results : 
figure;
xx = linspace(-0.25 , 0.25 , 1000);
plot(xx , (sin(alph)/cos(alph))*xx )
hold on ;

plot(VA(1,ind), VA(2,ind), 'y.');
plot(VB(1,ind), VB(2,ind), 'b.');
plot(VC(1,ind), VC(2,ind), 'r.');
j= 1;
for i = ind 
    
    plot([VA(1,i) VB(1,i)],[VA(2,i) VB(2,i)],'g');
    plot([VA(1,i) VC(1,i)],[VA(2,i) VC(2,i)],'g');
    plot([VB(1,i) VC(1,i)],[VB(2,i) VC(2,i)],'g');
    plot(point_min_dist(1,j),point_min_dist(2,j),'rx');
    plot([VA(1,i) point_min_dist(1,j)],[VA(2,i) point_min_dist(2,j)],'r');
    j=j+1;
end
xlim(Output.xran); ylim(Output.yran);


% Test if it's still work in the reference element : 
for j = 1:20
    
    VA_ref = transformation_phy_to_ref(VA(:,ind(j)) ,Ainv(:,ind(j)),b(:,ind(j))) ;
    VB_ref = transformation_phy_to_ref(VB(:,ind(j)) ,Ainv(:,ind(j)),b(:,ind(j))) ;
    VC_ref = transformation_phy_to_ref(VC(:,ind(j)) ,Ainv(:,ind(j)),b(:,ind(j))) ;

    x_inter = transformation_phy_to_ref(point_intersection(:,j),Ainv(:,ind(j)),b(:,ind(j)));
    x_min_dist = transformation_phy_to_ref(point_min_dist(:,j), Ainv(:,ind(j)) , b(:,ind(j)));
    angle_ref = angl_ref(ind(j))

    T11 = -sin(angle_ref);
    T12 = -cos(angle_ref);
    T21 = cos(angle_ref) ;
    T22 = -sin(angle_ref);
    determinant_T = T11.*T22 - T12.*T21;
    Tinv11 = T22./determinant_T;
    Tinv12 = -T12./determinant_T;
    Tinv21 = -T21./determinant_T;
    Tinv22 = T11./determinant_T;

    dir_perp = [ -sin(angle_ref) ; cos(angle_ref) ] ;
    dist_min_1 = Tinv11*(x_inter(1)-VA_ref(1))+ Tinv12*(x_inter(2)-VA_ref(2)) ;
    dist_min_2 = Tinv21*(x_inter(1)-VA_ref(1))+ Tinv22*(x_inter(2)-VA_ref(2)) ;
    %point_min_dist_ = VA_ref  + dist_min_1*dir_perp ;
    
    dir = [ cos(angle_ref) ; sin(angle_ref) ] ;
    dist_min = -x_inter(1)*cos(angle_ref) - x_inter(2)*sin(angle_ref) + VA_ref(1)*cos(angle_ref) + VA_ref(2)*sin(angle_ref) ;
    point_min_dist_ = x_inter  + dist_min*dir ;


    figure;
    lam = linspace(-3 , 3 , 1000);
    dist = linspace(-1,1,4);
    level_0 = [ -1 1 ];
    level_1 = linspace(-1,dist(2),2);
    level_2 = linspace(-1,dist(3),3);
    plot(x_inter(1) + lam*cos(angle_ref) ,x_inter(2) + lam*(sin(angle_ref)) )
    hold on
    plot(level_0(1) ,level_0(2),'bo');
    plot(level_1, [dist(3) dist(3)],'bo');
    plot(level_2 , [ dist(2) dist(2) dist(2) ],'bo');
    plot(dist , [ dist(1) dist(1) dist(1) dist(1) ],'bo');

    plot(VA_ref(1), VA_ref(2), 'b.');
    plot(VB_ref(1), VB_ref(2), 'b.');
    plot(VC_ref(1), VC_ref(2), 'b.');
    plot([VA_ref(1) VB_ref(1)],[VA_ref(2) VB_ref(2)],'g');
    plot([VA_ref(1) VC_ref(1)],[VA_ref(2) VC_ref(2)],'g');
    plot([VB_ref(1) VC_ref(1)],[VB_ref(2) VC_ref(2)],'g');
    plot(x_inter(1),x_inter(2),'rx');
    plot(point_min_dist_(1),point_min_dist_(2),'rx');
    plot([VA_ref(1) point_min_dist_(1)],[VA_ref(2) point_min_dist_(2)],'r');
    plot(x_min_dist(1),x_min_dist(2),'yx');
    %plot([VA_ref(1) x_min_dist(1)],[VA_ref(2) x_min_dist(2)],'y');
    xlim([-1.5 1.5]);
    ylim([-1.5 1.5]);
end



% Plot angle
figure;
quiver(xavg,yavg,cos(angle_phy_check),sin(angle_phy_check),'r');
xlim(Output.xran); ylim(Output.yran);

% Value on boundary from neighboring elements
Mesh.vmapM = reshape(Mesh.vmapM, Mesh.Nfp*Mesh.Nfaces, Mesh.K); 
Mesh.vmapP = reshape(Mesh.vmapP, Mesh.Nfp*Mesh.Nfaces, Mesh.K);
QP = Q(Mesh.vmapP);

% Construct training set
train_in = [Q(:,ind); QP(:,ind)];
train_out = angl_ref(ind);

% Export the training set 
data = [Q(:,ind)' , angl_ref(ind)' ]
csvwrite('data_test.csv',data)