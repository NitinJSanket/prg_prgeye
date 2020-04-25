clc
clear
close all
warning off

%% Up board TFlite performance is mostly less than 1 Hz in all networks.
%  Choose a RowNum from the following Network
% VanillaNetFloat: 1, VanillaNetInt: 8;
% ResNetFloat: 15, ResNetInt: 21;
% SqueezeNetFloat: 28, SqueezeNetInt: 34;
% MobileNetFloat: 42, MobileNetInt: 49;
% ShuffleNet: 57;
RowNum = 38;

switch RowNum
    case 1
        Network = 'VanillaNetSmall';
        Origin = [0.5, 2, 1e-5];
    case 5
        Network = 'VanillaNet';
        Origin = [0.5, 0.8, 1e-5];
    case 10
        Network = 'ResNetSmall';
        Origin = [0.5, 2, 1e-5];
    case 14
        Network = 'ResNet';
        Origin = [0.5, 0.5, 1e-5];
    case 19
        Network = 'SqueezeNetSmall';
        Origin = [0.5, 2, 1e-5];
    case 23
        Network = 'SqueezeNet';
        Origin = [0.5, 0.1, 1e-5];
    case 28
        Network = 'MobileNetSmall';
        Origin = [0.5, 3, 1e-5];
    case 32
        Network = 'MobileNet';
        Origin = [0.5, 1.0, 1e-5];
    case 37
        Network = 'ShuffleNetSmall';
        Origin = [0.5, 0.7, 1e-5];
    case 38
        Network = 'ShuffleNet';
        Origin = [0.5, 3, 1e-5];
    otherwise
        error('Incorrect RowNumber for Network');
end


disp(['Current Network: ', Network]);

Weights = [80, 136, 20, 7.2, 15, 7.2, 200, 300, 300, 300, 27.2, 250, 250, 22];
IdxsMakeNaN = [6, 14];
Weights(IdxsMakeNaN) = NaN;

% Laptop weight is 2220 and PC weight is 15000
Volume =  [96.3, 116.2, 15.6, 4.8, 9.75, 4.8, 208.8, 59192, 59192, 59192, 20.4, 3170.91, 3170.91, 17.2]'; % cm^3
Volume = (log(Volume) - 1.2) * 1000;
% 59192.6 is the volume of Desktop PC
% 544 is the largest dimension of Desktop PC
% 1223.898 is the volume of NUC and 901 is the weight

LargestDimension = [86, 88, 65, 40, 40, 40, 87, 560, 560, 560, 391, 391, 53]';
Names = {'Up', 'CoralDev', 'CoralUSB', 'NanoPi', 'BananaPiM2-Zero',...
    'NanoPi@0.48GHz', 'TX2', 'PC-i9', 'PC-i9NoAVX', 'PC-TitanXp', 'NanoPi-Coral', 'Laptop-i7', 'Laptop-1070', 'MaixBit'};
% Dark Blue, Light Green, Dark Green, Light Red, Medium Red, Dark Red,
% Yellow, Dark Gray, Medium Gray, Light Gray, Orange

T = readtable('SpeedTestWithNUC.xlsx');

% Add Names for each
% Color for each
FaceColor =    [...
    102, 191, 255;    % Blue
    229, 231, 88;     % Yellow
    116, 229, 90;     % Light Green
    234, 161, 233;    % Pink
    255, 143, 34;     % Orange
    255, 120, 120;    % Red
    180, 180, 180;    % Gray
    136, 116, 182;    % Cyan
    100, 100, 100;    % Black
    179, 143, 129;    % Brown
    067, 209, 147;    % Jungle Green
    255, 120, 120;    % Red
    160, 160, 0;      % Teal
    160, 160, 0;  % Teal
    ]./255;

figure('units','normalized','outerposition',[0 0 1 1]),
hold on;

nNetworks = 3;
if RowNum >= 37
    nNetworks = 1;
end


for count2 = 1:nNetworks
    
    ColNum = 2:14;
    FPSCell = table2cell(T(RowNum+count2, ColNum));
    for count3 = 1:length(FPSCell)
        if(~isa(FPSCell{count3},'double'))
            FPS(count3) = str2double(FPSCell{count3});
        else
            FPS(count3) = FPSCell{count3};
        end
    end
    FPS(end+1) = NaN; % MaixBit
    % F32, I8TFLite, I8EdgeTPU
    EdgeColor = {'r', 'k', 'b'};
    
    for count = 1:length(Names)
        Scatter1 = scatter(Weights(count), FPS(count), Volume(count), ...
            'MarkerFaceColor', FaceColor(count, :),...
            'MarkerEdgeColor', EdgeColor{count2}, 'LineWidth', 2.5);
    end
end


for count2 = 1:nNetworks
    
    ColNum = 2:14;
    FPSCell = table2cell(T(RowNum+count2, ColNum));
    for count3 = 1:length(FPSCell)
        if(~isa(FPSCell{count3},'double'))
            FPS(count3) = str2double(FPSCell{count3});
        else
            FPS(count3) = FPSCell{count3};
        end
    end
    FPS(end+1) = NaN; % MaixBit
    % F32, I8TFLite, I8EdgeTPU
    
    for count = 1:length(Names)
        scatter(Weights(count), FPS(count), 10, 'MarkerFaceColor', 'w', ...
            'MarkerEdgeColor', 'w');
        
    end
end


scatter(Origin(1), Origin(2), Origin(2), 'MarkerFaceColor', 'w',...
    'MarkerEdgeColor', 'w');

title(Network);
xlabel('Weight');
ylabel('FPS');
set(gca,'yscale','log');
% set(gca,'xscale','log');
set(gca, 'GridAlpha', 0.5);
set(gca, 'MinorGridAlpha', 0.5);
set(gca, 'FontSize', 24);
set(gca, 'FontName', 'Roboto');
% set(gca, 'linewidth', 6);
axis on;
grid on;
grid minor;
axis([0, 325, 0, 2000]);
ax = gca;
ax.XAxis.MinorTick = 'on';
ax.XAxis.MinorTickValues = ax.XAxis.Limits(1):4:ax.XAxis.Limits(2);
% ax.YAxis.MinorTickValues = ax.YAxis.Limits(1):10:ax.YAxis.Limits(2);
saveas(gcf, [Network, '.eps'], 'epsc');
saveas(gcf, [Network, '.png']);