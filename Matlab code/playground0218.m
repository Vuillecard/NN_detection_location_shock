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
alph=60*pi/180;
InitialCond = @(x,y) -1.7*(y<tan(alph)*(x))+1.33*(y>=tan(alph)*x);

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

% Entries of the matrix representing the geometric mapping
A11 = 1/2*(VB(1,:)-VA(1,:));
A12 = 1/2*(VC(1,:)-VA(1,:));
A21 = 1/2*(VB(2,:)-VA(2,:));
A22 = 1/2*(VC(2,:)-VA(2,:));

% Repeat with inverse
determinant = A11.*A22 - A12.*A21;
Ainv11 = A22./determinant;
Ainv12 = -A12./determinant;
Ainv21 = -A21./determinant;
Ainv22 = A11./determinant;

% Set angle
angl_phy(ind) = alph;
angl_phy(setdiff(1:Mesh.K,ind)) = NaN;

% Map from physical to reference
angl_ref = atan2(Ainv21.*cos(angl_phy)+Ainv22.*sin(angl_phy),Ainv11.*cos(angl_phy)+Ainv12.*sin(angl_phy));

% Map from reference to physical
angle_phy_check = atan2(A21.*cos(angl_ref)+A22.*sin(angl_ref),A11.*cos(angl_ref)+A12.*sin(angl_ref));

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