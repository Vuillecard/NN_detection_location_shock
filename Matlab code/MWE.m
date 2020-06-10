cd ./deepray-dgann
mypath
cd ..

CleanUp2D;

close all; clear all; clc;

%% SET PARAMETERS

% boundary conditions
BC_cond = {100001,'P'; 100002,'P'; 100003,'P'; 100004,'P'};

%solution to be considered
alph=73*pi/180;
InitialCond = @(x,y) 1*(y<tan(alph)*x)+3*(y>=tan(alph)*x);

%polynomial order
N = 3;

%ranges to visualize solution
xran = [-1,1]; yran = [-1,1];

% Mesh file
% generate using >> /Applications/Gmsh.app/Contents/MacOS/gmsh <geo_filename>.geo -2 -format msh2 -o <mesh_filename>.msh 
msh_file = 'mesh_square_trans.msh';

%% RUN THE CONSTRUCTOR (SHOULD NOT MATTER HERE)

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
    
%% APPLY THE NETWORK

% Evaluate solution
Q = feval(Problem.InitialCond, Mesh.x, Mesh.y);

% Boundary conditions
QG = ApplyBCScalar2D(Q,0,Mesh);

% Evaluate the network
nconst_ind = FindNonConstCells2D(Q);
ind = NN_Indicator2D(Q,QG,Mesh,Net,nconst_ind);

%% VISUALIZE THE RESULTS

% Plot Mesh and troubled cells
xavg = Mesh.AVG2D*Mesh.x;
yavg = Mesh.AVG2D*Mesh.y;
figure;
plot(xavg(ind), yavg(ind), 'rx'); 
hold on;
plot(xavg(setdiff(1:length(xavg),ind)), yavg(setdiff(1:length(yavg),ind)), 'b.'); 
xlim(Output.xran); ylim(Output.yran);
