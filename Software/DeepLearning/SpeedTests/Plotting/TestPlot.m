clc
clear all
close all
warning off

% Names = {'NanoPi1', 'NanoPi2', 'NanoPi3', 'MaixBit', 'CoralDev', 'CoralAcc', 'TX2', 'Up', 'PC1', 'PC2', 'PC3'};
% LargestDimension = [40, 40, 40, 53, 88, 65, 87, 86, 544, 544, 544]';
% Volume =  [24, 24, 24, 17.2, 116.2, 15.6, 65.3, 96.3, 59192.6, 59192.6, 59192.6]'; % cm^3
% Weights = [7.2, 7.2, 7.2, 22, 136, 20, 200, 100, 100, 100, 100];
% FPS = [34.3, 29.4, 17.0, 30.0, nan, nan, 232.3, 10.3, 920.8, 355.7, 556.7];

Weights = [100, 136, 20, 7.2, 7.2, 7.2, 200, 901, 901, 100, 22];
Volume =  [96.3, 116.2, 15.6, 24, 24, 24, 65.3, 1223.898, 1223.898, 59192.6/1000, 17.2]'; % cm^3
% 59192.6 is the volume of Desktop PC
% 544 is the largest dimension of Desktop PC

LargestDimension = [86, 88, 65, 40, 40, 40, 87, 221, 221, 221, 53]';
Names = {'Up', 'CoralDev', 'CoralAcc', 'NanoPi1', 'NanoPi2', 'NanoPi3', 'TX2', 'PC1', 'PC2', 'PC3', 'MaixBit'};
% Dark Blue, Light Green, Dark Green, Light Red, Medium Red, Dark Red,
% Yellow, Dark Gray, Medium Gray, Light Gray, Orange

% NanoPi1: 1.368GHz,  NanoPi2: 1.008GHz, NanoPi3: 480MHz
% PC1: i9 CPU with AVX, PC2: i9 CPU without AVX, PC3: Titan Xp
T = readtable('SpeedTest.xlsx');

RowNum = 4;
ColNum = 2:12;
FPS = str2double(table2cell(T(RowNum, ColNum)));
FPS(end) = 30.0;
% FPS = [10.30460276, nan, nan, 34.30157519, 29.36074927, 16.98266618, 232.2962543, 920.8050862, 355.6707308, 556.7078836, 30.0];


% Color for each
Color = [0, 0, 1;
    0, 0.8, 0;
    0, 0.4, 0;
    1, 0, 0;
    0.7, 0.2 ,0.2;
    0.5, 0, 0;
    0.9, 0.9 ,0;
    0.2, 0.2 ,0.2;
    0.5, 0.5 ,0.5;
    0.7, 0.7 ,0.7;
    1, 0.5 ,0];

figure,
hold on;

for count = 1:length(Names)
    Scatter1 = scatter(FPS(count), Weights(count), Volume(count)*100, 'MarkerFaceColor',...
        Color(count, :), 'MarkerEdgeColor', Color(count, :));
    scatter(FPS(count), Weights(count), 10, 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k');
    Scatter1.MarkerFaceAlpha = 0.4;
    Scatter1.MarkerEdgeAlpha = 0.4; 
end
xlabel('FPS');
ylabel('Weight');
set(gca,'xscale','log');
axis on;
grid on;
grid minor;