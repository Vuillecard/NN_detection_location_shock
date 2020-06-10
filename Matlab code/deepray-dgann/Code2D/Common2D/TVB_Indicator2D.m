function ind = TVB_Indicator2D(Q,QG,Mesh,TVBM,TVBnu)

% Purpose: find all the troubled-cells for variable Q using TBV-minmod 

eps = 1e-10;

% Find neighbors in patch
E1 = Mesh.EToE(:,1)'; E2 = Mesh.EToE(:,2)'; E3 = Mesh.EToE(:,3)';

% Extracting linear part of solution for elements. 
% We only keep the modes 1,2 and N+2
if(Mesh.N==1)
    Ql  = Q;
    QGl = QG;
else
    Qm              = Mesh.invV*Q;
    killmodes       = [3:Mesh.N+1,Mesh.N+3:Mesh.Np];
    Qm(killmodes,:) = 0;
    Ql              = Mesh.V*Qm;
    
    QGm              = Mesh.invV*QG;
    killmodes        = [3:Mesh.N+1,Mesh.N+3:Mesh.Np];
    QGm(killmodes,:) = 0;
    QGl              = Mesh.V*QGm;
end

% Get cell averages of patch. For the time being, Neuman BC used (except 
% for periodic BC)
AVG0 = Mesh.AVG2D*Ql;
AVGn = [AVG0(E1); AVG0(E2); AVG0(E3)];

% Replacing boundary element neighbours with ghost neighbours
AVGGE = Mesh.AVG2D*QGl;
GE1 = find(Mesh.EToGE(:,1))';
GE2 = find(Mesh.EToGE(:,2))';
GE3 = find(Mesh.EToGE(:,3))';
AVGn(1,GE1) = AVGGE(Mesh.EToGE(GE1,1)); 
AVGn(2,GE2) = AVGGE(Mesh.EToGE(GE2,2)); 
AVGn(3,GE3) = AVGGE(Mesh.EToGE(GE3,3));

% xavg = AVG2D*x;
% yavg = AVG2D*y;
% 
% figure(100)
% subplot(2,2,1)
% scatter3(xavg,yavg,AVG0,'o')
% subplot(2,2,2)
% scatter3(xavg,yavg,AVG0,'o')
% subplot(2,2,3)
% scatter3(xavg,yavg,AVG0,'o')
% subplot(2,2,4)
% scatter3(xavg,yavg,AVG0,'o')



% Get value at face mid point value for linear projection
QCF1 = Mesh.facemid1*Ql(Mesh.Fmask(:,1),:); 
QCF2 = Mesh.facemid2*Ql(Mesh.Fmask(:,2),:); 
QCF3 = Mesh.facemid3*Ql(Mesh.Fmask(:,3),:);

% Qtildes
Qtilde1 = QCF1 - AVG0; Qtilde2 = QCF2 - AVG0; Qtilde3 = QCF3 - AVG0;

% DelQs
ind1    = reshape(Mesh.patch_alphas(1,3,:),[1,Mesh.K]); ind1xy  = sub2ind(size(AVGn),ind1,1:Mesh.K);
ind2    = reshape(Mesh.patch_alphas(2,3,:),[1,Mesh.K]); ind2xy  = sub2ind(size(AVGn),ind2,1:Mesh.K);
ind3    = reshape(Mesh.patch_alphas(3,3,:),[1,Mesh.K]); ind3xy  = sub2ind(size(AVGn),ind3,1:Mesh.K);
DelQ1   = sum( reshape(Mesh.patch_alphas(1,1:2,:),[2,Mesh.K]).* ...
               [AVGn(1,:) - AVG0; AVGn(ind1xy) - AVG0] );
DelQ2   = sum( reshape(Mesh.patch_alphas(2,1:2,:),[2,Mesh.K]).* ...
               [AVGn(2,:) - AVG0; AVGn(ind2xy) - AVG0] );
DelQ3   = sum( reshape(Mesh.patch_alphas(3,1:2,:),[2,Mesh.K]).* ...
               [AVGn(3,:) - AVG0; AVGn(ind3xy) - AVG0] );           

% Get limited slopes/modes           
s1 = minmodB([Qtilde1;TVBnu*DelQ1],TVBM,Mesh.dx);
s2 = minmodB([Qtilde2;TVBnu*DelQ2],TVBM,Mesh.dx);
s3 = minmodB([Qtilde3;TVBnu*DelQ3],TVBM,Mesh.dx);

% Fix slope if s1+s2+s3 = 0
ind0   = find(abs(s1+s2+s3)>eps);
% length(ind0)
pos    = max(0,s1(ind0)) + max(0,s2(ind0)) + max(0,s3(ind0));
neg    = max(0,-s1(ind0)) + max(0,-s2(ind0)) + max(0,-s3(ind0));
thetap = min(1,neg./pos); 
thetam = min(1,pos./neg);
s1_fix = thetap.*max(0,s1(ind0)) - thetam.*max(0,-s1(ind0));
s2_fix = thetap.*max(0,s2(ind0)) - thetam.*max(0,-s2(ind0));
s3_fix = thetap.*max(0,s3(ind0)) - thetam.*max(0,-s3(ind0));
% max(abs(s1_fix - s1(ind0)))
% max(abs(s2_fix - s2(ind0)))
% max(abs(s3_fix - s3(ind0)))
s1(ind0) = s1_fix;
s2(ind0) = s2_fix;
s3(ind0) = s3_fix;


% Get limited face mid-point values
QCFL1 = AVG0 + s1; QCFL2 = AVG0 + s2; QCFL3 = AVG0 + s3;

% Mark cells where even one of the face values has changed 
% QCF1_true = facemid1*Q(Fmask(:,1),:);
% QCF2_true = facemid2*Q(Fmask(:,2),:); 
% QCF3_true = facemid3*Q(Fmask(:,3),:);


% for i = 1:K
%     figure(1001)
%     plot(0,QCF1_true(i),'o',0,QCF1(i),'x',0,QCFL1(i),'s')
%     
%     figure(1002)
%     plot(0,QCF2_true(i),'o',0,QCF2(i),'x',0,QCFL2(i),'s')
%     
%     figure(1003)
%     plot(0,QCF3_true(i),'o',0,QCF3(i),'x',0,QCFL3(i),'s')
%     
%     pause(0.1)
% end

%ind = find(abs(QCFL1 - QCF1_true) > eps | abs(QCFL2 - QCF2_true) > eps | abs(QCFL3 - QCF3_true) > eps);
ind = find(abs(QCFL1 - QCF1) > eps | abs(QCFL2 - QCF2) > eps | abs(QCFL3 - QCF3) > eps);



return
