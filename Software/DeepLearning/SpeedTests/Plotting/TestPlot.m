clc
clear all
close all
warning off
 
% Names = {'NanoPi1', 'NanoPi2', 'NanoPi3', 'MaixBit', 'CoralDev', 'CoralAcc', 'TX2', 'Up', 'PC1', 'PC2', 'PC3'};
% LargestDimension = [40, 40, 40, 53, 88, 65, 87, 86, 544, 544, 544]';
% Volume =  [24, 24, 24, 17.2, 116.2, 15.6, 65.3, 96.3, 59192.6, 59192.6, 59192.6]'; % cm^3
% Weights = [7.2, 7.2, 7.2, 22, 136, 20, 200, 100, 100, 100, 100];
% FPS = [34.3, 29.4, 17.0, 30.0, nan, nan, 232.3, 10.3, 920.8, 355.7, 556.7];
% FPS = [10.30460276, nan, nan, 34.30157519, 29.36074927, 16.98266618, 232.2962543,...
% 920.8050862, 355.6707308, 556.7078836, 30.0];
 
 
%% Choose a RowNum from the following Network:
% VanillaNetFloat: 1, VanillaNetInt: 8;
% ResNetFloat: 15, ResNetInt: 21;
% SqueezeNetFloat: 28, SqueezeNetInt: 34;
% MobileNetFloat: 42, MobileNetInt: 49;
% ShuffleNet: 57;
 
RowNum = 15;
 
switch RowNum
    case 1
        Network = 'VanillaNetFloat';
    case 8
        Network = 'VanillaNetInt';
    case 15
        Network = 'ResNetFloat';
    case 21
        Network = 'ResNetInt';
    case 28
        Network = 'SqueezeNetFloat';
    case 34
        Network = 'SqueezeNetInt';
    case 42
        Network = 'MobileNetFloat';
    case 49
        Network = 'MobileNetInt';
    case 57
        Network = 'ShuffleNet';
    otherwise
        error('Incorrect RowNumber for Network');
end
 
 
disp(['Current Network: ', Network]);
 
Weights = [100, 136, 20, 7.2, 7.2, 7.2, 200, 400, 400, 400, 22];
WeightRatio = 0.3;
Weights = Weights * WeightRatio;
Volume =  [96.3, 116.2, 15.6, 24, 24, 24, 65.3, 59192, 59192, 59192, 17.2]'; % cm^3
% 59192.6 is the volume of Desktop PC
% 544 is the largest dimension of Desktop PC
% 1223.898 is the volume of NUC and 901 is the weight
 
LargestDimension = [86, 88, 65, 40, 40, 40, 87, 221, 221, 221, 53]';
Names = {' Up', ' CoralDev', ' CoralUSB', ' NanoPi@1.368GHz', ' NanoPi@1.008GHz',...
    ' NanoPi@0.48GHz', ' TX2', ' PC-i9', ' PC-i9NoAVX', ' PC-TitanXp', ' MaixBit'};
% Dark Blue, Light Green, Dark Green, Light Red, Medium Red, Dark Red,
% Yellow, Dark Gray, Medium Gray, Light Gray, Orange
 
% NanoPi1: 1.368GHz,  NanoPi2: 1.008GHz, NanoPi3: 480MHz
% PC1: i9 CPU with AVX, PC2: i9 CPU without AVX, PC3: Titan Xp
T = readtable('SpeedTest.xlsx');
 
% Add Names for each
% Color for each
FaceColor = [0, 0, 1;
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
 
figure('units','normalized','outerposition',[0 0 1 1]),
hold on;
 
nNetworks = 5;
if RowNum > 49
    nNetworks = 2;
end
 
 
for count2 = 1:nNetworks
    
    ColNum = 2:12;
    FPS = str2double(table2cell(T(RowNum+count2, ColNum)));
    FPS(end) = 30.0;
    EdgeColor = ['r', 'g', 'b', 'k', 'w'];
    
    for count = 1:length(Names)
        Scatter1 = scatter(Weights(count), FPS(count), log10(Volume(count))*1000, 'MarkerFaceColor',...
            FaceColor(count, :), 'MarkerEdgeColor', EdgeColor(count2), 'LineWidth', 1.5);
        scatter(Weights(count), FPS(count), 10, 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k');
        Scatter1.MarkerFaceAlpha = 0.3;
        Scatter1.MarkerEdgeAlpha = 0.6;
        text(Weights(count), FPS(count), Names{count});
    end
    hold on;
end
title(Network);
xlabel('Weight');
ylabel('FPS');
set(gca,'yscale','log');
set(gca,'yscale','log');
set(gca, 'GridAlpha', 1);
set(gca, 'MinorGridAlpha', 1);
axis on;
grid on;
grid minor;