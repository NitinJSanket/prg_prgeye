clc
clear
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

%% Up board TFlite performance is mostly less than 1 Hz in all networks.

RowNum = 37;

switch RowNum
    case 1
        Network = 'VanillaNetSmall';
        Origin = [0.5, 5, 1e-5];
    case 5
        Network = 'VanillaNet';
        Origin = [0.5, 5, 1e-5];
    case 10
        Network = 'ResNetSmall';
        Origin = [0.5, 3.5, 1e-5];
    case 14
        Network = 'ResNet';
        Origin = [0.5, 1, 1e-5];
    case 19
        Network = 'SqueezeNetSmall';
        Origin = [0.5, 3.5, 1e-5];
    case 23
        Network = 'SqueezeNet';
        Origin = [0.5, 0.2, 1e-5];
    case 28
        Network = 'MobileNetSmall';
        Origin = [0.5, 5, 1e-5];
    case 32
        Network = 'MobileNet';
        Origin = [0.5, 1.5, 1e-5];
    case 37
        Network = 'ShuffleNet';
        Origin = [0.5, 0.8, 1e-5];
    otherwise
        error('Incorrect RowNumber for Network');
end


disp(['Current Network: ', Network]);

Weights = ([100, 136, 20, 7.2, 15, 7.2, 200, 260, 260, 260, 27.2, 22]);
% Weights = erf(sqrt(pi)/2 * log10(Weights));
%exp(-Weights)./(1 + exp(-Weights));
% Weights = [100, 136, 20, 7.2, 15, 7.2, 200, NaN, NaN, NaN, 27.2, 22];
% Weights = [100, NaN, NaN, 7.2, 15, NaN, NaN, NaN, NaN, NaN, NaN, NaN];
WeightRatio = 1;
Weights = Weights * WeightRatio;
% Volume =  [96.3, 116.2, 15.6, 24, 24, 24, 65.3, 59192, 59192, 59192, 17.2]'; % cm^3
Volume =  [96.3, 116.2, 15.6, 4.8, 9.75, 4.8, 65.3, 59192, 59192, 59192, 20.4, 17.2]'; % cm^3
a = 2; b = 2; c = -6.9; d = -0.6;
% Volume = (Volume)*200;
Volume = (log(Volume) - 1.2) * 1000;
disp(Volume)
% Volume = a.*((b.*(Volume+c)).^(1/3) + d).*100;%Volume).^(1/3)*400;
% Volume = erf(sqrt(pi)/2 * log10(Volume)/10)*100;
% disp(Volume);
% 59192.6 is the volume of Desktop PC
% 544 is the largest dimension of Desktop PC
% 1223.898 is the volume of NUC and 901 is the weight

LargestDimension = [86, 88, 65, 40, 40, 40, 87, 221, 221, 221, 53]'; %NanoPi@1.008GHz
Names = {' Up', ' CoralDev', ' CoralUSB', ' NanoPi@1.368GHz', ' BananPiM2-Zero',...
    ' NanoPi@0.48GHz', ' TX2', ' PC-i9', ' PC-i9NoAVX', ' PC-TitanXp', ' NanoPi-Coral', ' MaixBit'};
% Dark Blue, Light Green, Dark Green, Light Red, Medium Red, Dark Red,
% Yellow, Dark Gray, Medium Gray, Light Gray, Orange

% NanoPi1: 1.368GHz,  NanoPi2: 1.008GHz, NanoPi3: 480MHz
% PC1: i9 CPU with AVX, PC2: i9 CPU without AVX, PC3: Titan Xp
T = readtable('SpeedTest.xlsx');

% Add Names for each
% Color for each
FaceColor =    [102, 191, 255;    % Blue
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
    160, 160, 0]   ...% Brown
    ./255;

figure('units','normalized','outerposition',[0 0 1 1]),
hold on;

nNetworks = 3;
if RowNum > 49
    nNetworks = 2;
end


for count2 = 1:nNetworks
    
    ColNum = 2:13;
    FPS = str2double(table2cell(T(RowNum+count2, ColNum)));
    FPS(end) = 30.0;
    % F32, I8TFLite, I8EdgeTPU
    EdgeColor = {'r', 'k', 'b'};%[0, 150, 200]./255};%, 'w', [255 38 252]/255};
    
    for count = 1:length(Names)
        if(count == 6)
            continue;
        end
%         Scatter1 = scatter(Weights(count), FPS(count),  Volume(count), 'o', 'MarkerFaceColor', FaceColor(count, :),...
%             'MarkerEdgeColor', 'w', 'LineWidth', 1);
          Scatter1 = scatter(Weights(count), FPS(count), Volume(count), ...
                                 'MarkerFaceColor', FaceColor(count, :),...
                            'MarkerEdgeColor', EdgeColor{count2}, 'LineWidth', 2.5);
        %         Scatter1.MarkerFaceAlpha = 0.3;
        %         Scatter1.MarkerEdgeAlpha = 0.6;
        %         text(Weights(count), FPS(count), Names{count});
    end
end


for count2 = 1:nNetworks
    
    ColNum = 2:13;
    FPS = str2double(table2cell(T(RowNum+count2, ColNum)));
    FPS(end) = 30.0;
    % F32, I8TFLite, I8EdgeTPU
    EdgeColor = {'r', 'k', 'b'};%[0, 150, 200]./255};%, 'w', [255 38 252]/255};
    
    for count = 1:length(Names)
        if(count == 6)
            continue;
        end
        
        %         Scatter1 = scatter(Weights(count), FPS(count), log10(Volume(count))...
        %                     *1000, 'MarkerFaceColor', FaceColor(count, :),...
        %                     'MarkerEdgeColor', EdgeColor{count2}, 'LineWidth', 2.5);
        scatter(Weights(count), FPS(count), 10, 'MarkerFaceColor', 'w', ...
            'MarkerEdgeColor', 'w');
        %         Scatter1.MarkerFaceAlpha = 0.3;
        %         Scatter1.MarkerEdgeAlpha = 0.6;
        %         text(Weights(count), FPS(count), Names{count});
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
axis([0, 300, 0, 1650]);
ax = gca;
ax.XAxis.MinorTick = 'on';
ax.XAxis.MinorTickValues = ax.XAxis.Limits(1):4:ax.XAxis.Limits(2);
% ax.YAxis.MinorTickValues = ax.YAxis.Limits(1):10:ax.YAxis.Limits(2);
saveas(gcf, [Network, '.eps'], 'epsc');
saveas(gcf, [Network, '.png']);
close all;
