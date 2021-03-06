cd ./deepray-dgann
mypath
cd ..

CleanUp2D;

close all; clear all; clc;

% boundary conditions
% If using an structure mesh use BC_cond with 'P' otherwise use 'D'

%BC_cond = {100001,'D'; 100002,'D'; 100003,'D'; 100004,'D'};
BC_cond = {100001,'P'; 100002,'P'; 100003,'P'; 100004,'P'};

%solution to be considered
alph=70*pi/180;
InitialCond = @(x,y) 2*(y<tan(alph)*x)+4*(y>=tan(alph)*x);

%polynomial order p
N = 4;

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

% Repeat with inverse
determinant = A11.*A22 - A12.*A21;
Ainv11 = A22./determinant;
Ainv12 = -A12./determinant;
Ainv21 = -A21./determinant;
Ainv22 = A11./determinant;

Ainv = [ Ainv11 ;Ainv12  ;Ainv21 ;Ainv22 ];


data = [] ;
disp('... Data generation ')
degrees = unifrnd(0,180,1,50);
space_1 = unifrnd(-2,2,1,5);
space_2 = unifrnd(-2,2,1,5);
for deg = degrees   
    for a = space_1
        for b = space_2
            if a~=b 
                
                alph=deg*pi/180;
                fun = @(x,y) a*(y<tan(alph)*x)+b*(y>=tan(alph)*x);
                
                %Need to select of the alternative:
                %   data_ = data_gen(alph ,fun,Net,Mesh,Ainv11,Ainv12,Ainv21,Ainv22 ,true,false );
                %   data_ = data_gen_close_neighborhood(alph ,fun,Net,Mesh,Ainv11,Ainv12,Ainv21,Ainv22 ,true,false );
                %   data_ = data_gen_all_neighborhood_orthogonal(alph,fun,Net,Mesh,Ainv,b_,true);
                
                data_ = data_gen_all_neighborhood(alph ,fun,Net,Mesh,Ainv11,Ainv12,Ainv21,Ainv22 ,true,false );
                
                data = [data ; data_ ] ;
           end

       end
   end

    
       
end

% Save the data in a file 
csvwrite('\data\train_set\discontinuity_R\mesh_B\data_set_p_4_all.csv',data)
disp('... Data generation succeed ')


