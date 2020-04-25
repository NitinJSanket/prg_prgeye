clc
clear all
close all
warning off;

% Min. Acc 2.5 px.
% Names = {' Up', ' CoralDev', ' CoralUSB', ' NanoPi@1.368GHz', ' BananPiM2-Zero',...
%     ' NanoPi@0.48GHz', ' TX2', ' PC-i9', ' PC-i9NoAVX', ' PC-TitanXp', ' NanoPi-Coral', ' MaixBit'};
Names = {'Up', 'CoralDev', 'CoralUSB', 'NanoPi', 'BananaPiM2-Zero',...
    'NanoPi@0.48GHz', 'TX2', 'PC-i9', 'PC-i9NoAVX', 'PC-TitanXp',...
    'NanoPi-Coral', 'Laptop-i7', 'Laptop-1070', 'MaixBit'};
Config = {'RNSF32', 'RNSI8TPU', 'RNI8TPU', 'RNSI8', 'RNSI8', 'RNSI8', 'SNSF32', 'SNSF32',...
    'VNF32', 'SNSF32', 'RNI8TPU', 'SNSF32', 'SNSF32', 'None'};
FPS = [7.964729251, 1111, 847, 30.84564631, 23.99833935, 12.20463235, 224.3856457, 831.0621131, NaN,...#234.4692959
    978.0374911, 223.7100557, 264.7374341, 932.2498622, NaN];


Volume =  [96.3, 116.2, 15.6, 4.8, 9.75, 4.8, 208.8, 59192, 59192, 59192, 20.4, 3170.91, 3170.91, 17.2]'; % cm^3
Volume = (log(Volume) - 1.2) * 1000;
Weights = [80, 136, 20, 7.2, 15, 7.2, 200, 300, 300, 300, 27.2, 250, 250, 22];
IdxsMakeNaN = [6, 14];
Weights(IdxsMakeNaN) = NaN;

% Origin = [0, 300, 2, 2e3];

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


for count = 1:length(FPS)
    Scatter1 = scatter(Weights(count), FPS(count), Volume(count), ...
                                 'MarkerFaceColor', FaceColor(count, :),...
                            'MarkerEdgeColor', 'none', 'LineWidth', 2.5);
end

for count = 1:length(FPS)
    scatter(Weights(count), FPS(count), 10, 'MarkerFaceColor', 'w', ...
            'MarkerEdgeColor', 'w');
end

Origin = [0.5, 5, 1e-5];
scatter(Origin(1), Origin(2), Origin(2), 'MarkerFaceColor', 'w',...
    'MarkerEdgeColor', 'w');

% title('Best Network On Each Computer');
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

saveas(gcf, 'CompileDiffBoard.eps', 'epsc');
saveas(gcf, 'CompileDiffBoard.png');