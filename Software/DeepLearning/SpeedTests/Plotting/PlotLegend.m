clc
clear all
close all

Names = {'Up', 'CoralDev', 'CoralUSB', 'NanoPi', 'BananaPiM2-Zero',...
    'NanoPi@0.48GHz', 'TX2', 'PC-i9', 'PC-i9NoAVX', 'PC-TitanXp',...
    'NanoPi-Coral', 'Laptop-i7', 'Laptop-1070', 'MaixBit'};

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


for count = 1:length(Names)
    if(count > 7)
        Y = 0;
        count1 = count-7;
    else
        Y = 1;
        count1 = count;
    end
   Scatter1 = scatter(count1*2, Y, 1000, ...
            'MarkerFaceColor', FaceColor(count, :),...
            'MarkerEdgeColor', 'none', 'LineWidth', 2.5); 
        
        text(count1*2, Y, Names{count}, 'FontName', 'Roboto', 'FontSize', 24);
end


set(gca, 'FontSize', 24);
set(gca, 'FontName', 'Roboto');

saveas(gcf, 'Legend.eps', 'epsc');
saveas(gcf, 'Legend.png');
