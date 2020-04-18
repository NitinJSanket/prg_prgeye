clc
clear all
close all
warning off;

T = readtable('SpeedTest.xlsx', 'Sheet', 'VanillaNetDiffSizes');

Names = {'CoralUSB', 'CoralNanoPi', 'CoralDev', 'TX2', 'PC-i9', 'PC-TitanXp'};

RowsDeep = 3:8;
RowsWide = 11:17;
RowsWideAndDeep = 21:26;

XDeep = 2:7;
XWide = [20, 48, 64, 96, 128, 256, 512];
XWideAndDeep = [2.5, 25, 100, 250, 500, 1000];


FaceColor = [116, 229, 90;     % Light Green
    067, 209, 147;    % Jungle Green
    229, 231, 88;     % Yellow
    180, 180, 180;    % Gray
    100, 100, 100;    % Black
    179, 143, 129;]   ...% Brown
    ./255;

% NOTE: Num. FLOPs is linear with Num. Params

%% Deep
AxisLim = [2e5, 4e8, 0.5, 3e3];
PlotStuff(T, RowsDeep, Names, FaceColor, AxisLim);
title('Deep');

% figure('units','normalized','outerposition',[0 0 1 1]),
% NumParam = cell2mat(table2cell(T(RowsDeep, 10)));
% NumFLOPs = cell2mat(table2cell(T(RowsDeep, 11)));
% plot(NumParam, NumFLOPs);

%% Wide
AxisLim = [2e5, 4e8, 0.5, 3e3];
PlotStuff(T, RowsWide, Names, FaceColor, AxisLim);
title('Wide');

%% Wide And Deep
AxisLim = [2e5, 4e8, 0.5, 3e3];
PlotStuff(T, RowsWideAndDeep, Names, FaceColor, AxisLim);
title('WideAndDeep');

%%
function PlotStuff(T, Rows, Names, FaceColor, AxisLim)
figure('units','normalized','outerposition',[0 0 1 1]),
hold on;
FPS = cell2mat(table2cell(T(Rows, 3:8)));
NumParam = cell2mat(table2cell(T(Rows, 10)));
% NumFLOPs = cell2mat(table2cell(T(Rows, 11)));

for count = 1:length(Names)
   plot(NumParam, FPS(:, count), 'MarkerFaceColor', FaceColor(count, :), 'LineWidth', 2);
end
legend(Names);
% for count = 1:length(NumParam)
%     for count2 = 1:length(Names)
%         Scatter1 = scatter(NumParam(count), FPS(count, count2), 1000, ...
%             'MarkerFaceColor', FaceColor(count2, :),...
%             'MarkerEdgeColor', 'none', 'LineWidth', 2.5);
%     end
% end
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
axis(AxisLim); % [2e5, 4e8, 0.5, 3e3]
ax = gca;
ax.XAxis.MinorTick = 'on';
end