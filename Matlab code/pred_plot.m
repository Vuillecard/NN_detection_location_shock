function [] = pred_plot(data ,pred , Mesh,A,Output,name_graph )
% Plot the results of a given prediction, plot the orientation of the
% discontinuity

% evaluate solution 
ind = data(:,1)';

Q = data(:,2:end-1)';

true_alpha_ref(ind) = data(:,end)';
true_alpha_ref(setdiff(1:Mesh.K,ind)) = NaN;

true_angle_phy = atan2(A(3,:).*cos(true_alpha_ref)+A(4,:).*sin(true_alpha_ref),A(1,:).*cos(true_alpha_ref)+A(2,:).*sin(true_alpha_ref));

pred_alpha_ref(ind) = pred';
pred_alpha_ref(setdiff(1:Mesh.K,ind)) = NaN;
pred_angle_phy = atan2(A(3,:).*cos(pred_alpha_ref)+A(4,:).*sin(pred_alpha_ref),A(1,:).*cos(pred_alpha_ref)+A(2,:).*sin(pred_alpha_ref));


% Plot Mesh and troubled cells
xavg = Mesh.AVG2D*Mesh.x;
yavg = Mesh.AVG2D*Mesh.y;

figure

t = tiledlayout(2,2);
nexttile
plot(xavg(ind), yavg(ind), 'rx'); 
hold on;
plot(xavg(setdiff(1:length(xavg),ind)), yavg(setdiff(1:length(yavg),ind)), 'b.'); 
xlim(Output.xran); ylim(Output.yran);
title('Trouble cells in red')
% Plot angle
nexttile
quiver(xavg,yavg,cos(true_angle_phy),sin(true_angle_phy),'r');
xlim(Output.xran); ylim(Output.yran);
title('true choc angle')
nexttile
quiver(xavg,yavg,cos(true_angle_phy),sin(true_angle_phy),'r');
hold on
quiver(xavg,yavg,cos(pred_angle_phy),sin(pred_angle_phy),'b');
xlim(Output.xran); ylim(Output.yran);
legend('true','predict')
title(' true vs predict choc angle')

nexttile
quiver(xavg,yavg,cos(pred_angle_phy),sin(pred_angle_phy),'b');
xlim(Output.xran); ylim(Output.yran);
title('predicted choc angle')

title(t,name_graph)

x= linspace(-(1/sqrt(2)),(1/sqrt(2)),1000);
figure
quiver(xavg,yavg,cos(pred_angle_phy),sin(pred_angle_phy),'b');
xlim(Output.xran); ylim(Output.yran);
title('predicted circular shock')


figure
quiver(xavg,yavg,cos(pred_angle_phy),sin(pred_angle_phy),'b');
hold on
plot(x , sqrt(0.5-x.^2),'r')
plot(x , -sqrt(0.5-x.^2),'r')
xlim(Output.xran); ylim(Output.yran);
title('zoom on predicted circular shock ')



end 
