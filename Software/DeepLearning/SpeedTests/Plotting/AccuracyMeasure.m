clc
clear
close all
warning off

T = readtable('SpeedTestWithNUC.xlsx');


%% Acc/Param
figure('units','normalized','outerposition',[0 0 1 1]),
hold on;


ColNum = 20;
RowNum = [2, 6, 11, 15, 20, 24, 29, 33, 38, 39];
AccPerParam = cell2mat(table2cell(T(RowNum, ColNum)));
AccPerParam = [AccPerParam(1:2:end), AccPerParam(2:2:end)];
[AccPerParamSmall, AccPerParamSmallIdxs] = sort(AccPerParam(:,1), 'ascend');
[AccPerParamLarge, AccPerParamLargeIdxs] = sort(AccPerParam(:,2), 'ascend');

bar(1:5, AccPerParamSmall);
bar(6:10, AccPerParamLarge);
NamesSmall = {'VN', 'RN', 'SqN', 'MN', 'ShN'};
NamesSmall = {NamesSmall{AccPerParamSmallIdxs}};
NamesLarge = {'VN', 'RN', 'SqN', 'MN', 'ShN'};
NamesLarge = {NamesLarge{AccPerParamLargeIdxs}};
set(gca,'xtick',[1:10],'xticklabel',[NamesSmall, NamesLarge]);
grid on;
grid minor;
set(gca, 'GridAlpha', 0.5);
set(gca, 'MinorGridAlpha', 0.5);
ylabel('Accuracy Per Kilo Param');
title('Acc/Param');
set(gca, 'FontSize', 24);
set(gca, 'FontName', 'Roboto');
saveas(gcf, 'AccPerParam.eps', 'epsc');
saveas(gcf, 'AccPerParam.png');

%% Acc/Op
figure('units','normalized','outerposition',[0 0 1 1]),
hold on;

ColNum = 21;
RowNum = [2, 6, 11, 15, 20, 24, 29, 33, 38, 39];
AccPerOp = cell2mat(table2cell(T(RowNum, ColNum)));
AccPerOp = [AccPerOp(1:2:end), AccPerOp(2:2:end)];
[AccPerOpSmall, AccPerOpSmallIdxs] = sort(AccPerOp(:,1), 'ascend');
[AccPerOpLarge, AccPerOpLargeIdxs] = sort(AccPerOp(:,2), 'ascend');

bar(1:5, AccPerOpSmall);
bar(6:10, AccPerOpLarge);
NamesSmall = {'VN', 'RN', 'SqN', 'MN', 'ShN'};
NamesSmall = {NamesSmall{AccPerOpSmallIdxs}};
NamesLarge = {'VN', 'RN', 'SqN', 'MN', 'ShN'};
NamesLarge = {NamesLarge{AccPerOpLargeIdxs}};
set(gca,'xtick',[1:10],'xticklabel',[NamesSmall, NamesLarge]);
grid on;
grid minor;
set(gca, 'GridAlpha', 0.5);
set(gca, 'MinorGridAlpha', 0.5);
ylabel('Accuracy Per Kilo OP');
title('Acc/Op');
set(gca, 'FontSize', 24);
set(gca, 'FontName', 'Roboto');
saveas(gcf, 'AccPerOP.eps', 'epsc');
saveas(gcf, 'AccPerOP.png');


%% Acc
figure('units','normalized','outerposition',[0 0 1 1]),
hold on;

ColNum = 22;
RowNum = [2, 6, 11, 15, 20, 24, 29, 33, 38, 39];
Acc = cell2mat(table2cell(T(RowNum, ColNum)));
Acc = [Acc(1:2:end), Acc(2:2:end)];
[AccSmall, AccSmallIdxs] = sort(Acc(:,1), 'ascend');
[AccLarge, AccLargeIdxs] = sort(Acc(:,2), 'ascend');

bar(1:5, AccSmall);
bar(6:10, AccLarge);
NamesSmall = {'VN', 'RN', 'SqN', 'MN', 'ShN'};
NamesSmall = {NamesSmall{AccSmallIdxs}};
NamesLarge = {'VN', 'RN', 'SqN', 'MN', 'ShN'};
NamesLarge = {NamesLarge{AccLargeIdxs}};
set(gca,'xtick',[1:10],'xticklabel',[NamesSmall, NamesLarge]);
grid on;
grid minor;
set(gca, 'GridAlpha', 0.5);
set(gca, 'MinorGridAlpha', 0.5);
ylabel('Accuracy (%)');
title('Acc');
set(gca, 'FontSize', 24);
set(gca, 'FontName', 'Roboto');
saveas(gcf, 'Acc.eps', 'epsc');
saveas(gcf, 'Acc.png');

%% Acc*FPS
% figure('units','normalized','outerposition',[0 0 1 1]),
% hold on;
% 
% ColNum = 22;
% RowNum = [4, 8, 13, 17, 22, 25, 29, 33, 38, 39];
% Acc = cell2mat(table2cell(T(RowNum, ColNum)));
% Acc = [Acc(1:2:end), Acc(2:2:end)];
% [AccSmall, AccSmallIdxs] = sort(Acc(:,1), 'ascend');
% [AccLarge, AccLargeIdxs] = sort(Acc(:,2), 'ascend');
% 
% bar(1:5, AccSmall);
% bar(6:10, AccLarge);
% NamesSmall = {'VanillaNet', 'ResNet', 'SqueezeNet', 'MobileNet', 'ShuffleNet'};
% NamesSmall = {NamesSmall{AccSmallIdxs}};
% NamesLarge = {'VanillaNet', 'ResNet', 'SqueezeNet', 'MobileNet', 'ShuffleNet'};
% NamesLarge = {NamesLarge{AccLargeIdxs}};
% set(gca,'xtick',[1:10],'xticklabel',[NamesSmall, NamesLarge]);
% grid on;
% grid minor;
% set(gca, 'GridAlpha', 0.5);
% set(gca, 'MinorGridAlpha', 0.5);
% title('Acc');