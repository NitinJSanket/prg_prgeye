clc
clear all
close all
warning off;

% Add Laptop
% Add NanoPi
% Add BananaPi

T = readtable('SpeedTest.xlsx', 'Sheet', 'VanillaNetDiffSizes');

Names = {'Up', 'CoralDev', 'CoralUSB', 'NanoPi', 'BananaPiM2-Zero',...
    'NanoPi@0.48GHz', 'TX2', 'PC-i9', 'PC-i9NoAVX', 'PC-TitanXp',...
    'NanoPi-Coral', 'Laptop-i7', 'Laptop-1070', 'MaixBit'};


RowsDeep = 3:8;
RowsWide = 11:17;
RowsWideAndDeep = 21:26;

XDeep = 2:7;
XWide = [20, 48, 64, 96, 128, 256, 512];
XWideAndDeep = [2.5, 25, 100, 250, 500, 1000];


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

% NOTE: Num. FLOPs is linear with Num. Params

%% Deep
AxisLim = [2e5, 2e8, 0.2, 3e3];
PlotStuff(T, RowsDeep, Names, FaceColor, AxisLim);
title('Deep');
saveas(gcf, 'Deep.eps', 'epsc');
saveas(gcf, 'Deep.png');

%% Wide
AxisLim = [2e5, 1.5e8, 0.05, 3e3];
PlotStuff(T, RowsWide, Names, FaceColor, AxisLim);
title('Wide');
saveas(gcf, 'Wide.eps', 'epsc');
saveas(gcf, 'Wide.png');

%% Wide And Deep
AxisLim = [2e5, 1e8, 0.5, 3e3];
PlotStuff(T, RowsWideAndDeep, Names, FaceColor, AxisLim);
title('WideAndDeep');
saveas(gcf, 'WideAndDeep.eps', 'epsc');
saveas(gcf, 'WideAndDeep.png');

%% Legend
figure('units','normalized','outerposition',[0 0 1 1]),
PlotStuff(T, RowsWideAndDeep, Names, FaceColor, AxisLim);
legend(Names, 'NumColumns', 6, 'Location', 'best');
saveas(gcf, 'LegendDiffSizes.eps', 'epsc');
saveas(gcf, 'LegendDiffSizes.png');

%%
function PlotStuff(T, Rows, Names, FaceColor, AxisLim)
figure('units','normalized','outerposition',[0 0 1 1]),
hold on;
FPSCell = table2cell(T(Rows, 3:16));
for count3 = 1:size(FPSCell,1)
    for count4 = 1:size(FPSCell,2)
        if(~isa(FPSCell{count3, count4},'double'))
            FPS(count3, count4) = str2double(FPSCell{count3, count4});
        else
            FPS(count3, count4) = FPSCell{count3, count4};
        end
    end
end

% FPS = cell2mat(table2cell(T(Rows, 3:16)));
NumParam = cell2mat(table2cell(T(Rows, 18)));
% NumFLOPs = cell2mat(table2cell(T(Rows, 11)));

for count = 1:length(Names)
    %    plot(NumParam, FPS(:, count), 'MarkerFaceColor', FaceColor(count, :), 'LineWidth', 2);
    plot(NumParam, FPS(:, count), 'Color', FaceColor(count, :), 'LineWidth', 2);
end
% legend(Names, 'NumColumns', 6);

xlabel('NumParam');
ylabel('FPS');
set(gca,'xscale','log');
set(gca,'yscale','log');
set(gca, 'GridAlpha', 0.5);
set(gca, 'MinorGridAlpha', 0.5);
set(gca, 'FontSize', 24);
set(gca, 'FontName', 'Roboto');
axis on;
grid on;
grid minor;
axis(AxisLim);
ax = gca;
ax.XAxis.MinorTick = 'on';
end