cd ./deepray-dgann
mypath
cd ..

CleanUp2D;

close all; clear all; clc;

% boundary conditions
BC_cond = {100001,'D'; 100002,'D'; 100003,'D'; 100004,'D'};

%solution to be considered
alph=70*pi/180;
InitialCond = @(x,y) 2*(y<tan(alph)*x)+4*(y>=tan(alph)*x);

%polynomial order
N = 3;

%ranges to visualize solution
xran = [-1,1]; yran = [-1,1];

% Mesh file
% generate using >> /Applications/Gmsh.app/Contents/MacOS/gmsh <geo_filename>.geo -2 -format msh2 -o <mesh_filename>.msh 
msh_file = 'C:\Users\pierr\Desktop\Projet de semestre I\Meshes\mesh_B_unstr.msh';


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


% setting different angle 
deg1 = 30*pi/180;
deg2 = 60*pi/180;
deg3 = 90*pi/180;

% setting a discontinuty 
discont_1 =@(x,y) -1.7*(y<tan(deg1)*x)+ 1.3*(y>=tan(deg1)*x);
discont_2 =@(x,y) -1.7*(y<tan(deg2)*x)+ 1.3*(y>=tan(deg2)*x);
discont_3 =@(x,y) -1.7*(y<tan(deg3)*(x))+ 1.3*(y>=tan(deg3)*(x));
discont_circle = @(x,y) -1.7*(abs(y)<sqrt(0.5-x.^2))+ 1.3*( abs(y) >= sqrt(0.5-x.^2));


% making test for data_set_6
data_test_6_30 = data_gen_all_neighborhood(deg1,discont_1,Net,Mesh,Ainv11,Ainv12,Ainv21,Ainv22,false,false);
data_test_6_60 = data_gen_all_neighborhood(deg2,discont_2,Net,Mesh,Ainv11,Ainv12,Ainv21,Ainv22,false,false);
data_test_6_90= data_gen_all_neighborhood(deg3,discont_3,Net,Mesh,Ainv11,Ainv12,Ainv21,Ainv22,false,false);
data_test_6_circle = data_gen_all_neighborhood(deg3,discont_circle,Net,Mesh,Ainv11,Ainv12,Ainv21,Ainv22,false,true);

csvwrite('C:\Users\pierr\Desktop\Projet de semestre I\data\test_set\discontinuity_R\mesh_B_unstr\data_test_set_p_3_all_30.csv',data_test_6_30)
csvwrite('C:\Users\pierr\Desktop\Projet de semestre I\data\test_set\discontinuity_R\mesh_B_unstr\data_test_set_p_3_all_60.csv',data_test_6_60)
csvwrite('C:\Users\pierr\Desktop\Projet de semestre I\data\test_set\discontinuity_R\mesh_B_unstr\data_test_set_p_3_all_90.csv',data_test_6_90)
csvwrite('C:\Users\pierr\Desktop\Projet de semestre I\data\test_set\discontinuity_R\mesh_B_unstr\data_test_set_p_3_all_circle.csv',data_test_6_circle)

