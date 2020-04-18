clc
clear all
close all
warning off
 
QuadrotorSize = [0, 75, 120, 160, 210, 360, 500, 650, 700]';
HoverPower = [0, 5.75, 20.8, 15, 22.8, 73.5, 156.7, 127, 0]';
HoverPowerEfficiency = [0, 2.70, 2.51, 4.65, 5.88, 3.74, 3.19, 7.68, 0]'./14;
TotalPower = [0, 33, 97.7, 74.5, 106.2, 318, 656.8, 603, 700]';
PowerArray = [TotalPower-HoverPower.*4, HoverPower.*4];
PowerArray(end, :) = 0;

% X, Y: Size vs Total Power
% Radius is proportional to HoverPowerEfficiency
% Pie Chart complete is Total Power. Blue part is CPU Power and Red
% Part is Motor Power
bubblepie(QuadrotorSize, TotalPower, HoverPowerEfficiency, PowerArray, [], {}, '', '', 0);
grid on;
grid minor;
set(gca, 'GridAlpha', 0.5);
set(gca, 'MinorGridAlpha', 0.5);
set(gca, 'FontSize', 24);
set(gca, 'FontName', 'Roboto');
xlabel('Size (mm)');
ylabel('Total Power (W)');
set(gcf, 'units', 'normalized');
set(gcf, 'outerposition',[0 0 1 1]);

saveas(gcf, 'Power.eps', 'epsc');
saveas(gcf, 'Power.png');
