

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
