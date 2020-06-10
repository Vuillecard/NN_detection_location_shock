cd ./deepray-dgann
mypath
cd ..

CleanUp2D;

close all; clear all; clc;

% boundary conditions
BC_cond = {100001,'P'; 100002,'P'; 100003,'P'; 100004,'P'};

%solution to be considered
alph=70*pi/180;
InitialCond = @(x,y) 2*(y<tan(alph)*x)+4*(y>=tan(alph)*x);

%polynomial order
N = 5;

%ranges to visualize solution
xran = [-1,1]; yran = [-1,1];

% Mesh file
% generate using >> /Applications/Gmsh.app/Contents/MacOS/gmsh <geo_filename>.geo -2 -format msh2 -o <mesh_filename>.msh 
msh_file = 'Mesh_C_0p025_P.msh';

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

b1 = 1/2*(VB(1,:)+ VC(1,:));
b2 = 1/2*(VB(2,:)+VC(2,:));

b_ = [b1 ; b2] ; 

% Entries of the matrix representing the geometric mapping
A11 = 1/2*(VB(1,:)-VA(1,:));
A12 = 1/2*(VC(1,:)-VA(1,:));
A21 = 1/2*(VB(2,:)-VA(2,:));
A22 = 1/2*(VC(2,:)-VA(2,:));

A = [A11;A12;A21;A22];


data_set_1 = csvread('C:\Users\pierr\Desktop\Projet de semestre I\data\test_set\discontinuity_R\mesh_C\data_test_all_p5_circle.csv');

pred_1 = csvread('C:\Users\pierr\Desktop\Projet de semestre I\NN_pyhton_code\Task 2\data\pred_task_2data_test_all_p5_circle.csv');

pred_plot_ortho(data_set_1 ,pred_1, Mesh,A,Output,b_);
%pred_plot(data_set_1 ,pred_1 , Mesh,A,Output,'plots' )
