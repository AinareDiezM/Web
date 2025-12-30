clc; clear; close all;

% --- Folders ---
folderGT = 'C:\Users\Usuario\Desktop\BIALYSTOK\SEGMENTACIONES\Comparacion metodos\GT\GT_RAS_PNG_RECORTE';
folderSeg = 'C:\Users\Usuario\Desktop\BIALYSTOK\SEGMENTACIONES\Comparacion metodos\Todoslosmetodos - EDITAR MANUAL';

% --- List files ---
filesGT = dir(fullfile(folderGT, '*.png'));  
filesSeg = dir(fullfile(folderSeg, '*.png')); 

% --- Initialize results ---
results = table('Size',[length(filesSeg), 4], ...
                'VariableTypes',{'string','string','double','double'}, ...
                'VariableNames',{'Image','Method','Jaccard','Dice'});

% --- Loop over segmented images ---
for k = 1:length(filesSeg)
    segName = filesSeg(k).name;
    
    Iseg = imread(fullfile(folderSeg, segName));
    if ~islogical(Iseg)
        Iseg = Iseg > 0; 
    end
    
    tokens = split(segName, '_');
    ID = strjoin(tokens(1:2), '_'); 
    method = tokens{4};
    
    ID_GT = [ID '_GT_recortado'];  
    gtNames = erase({filesGT.name}, '.png'); 
    gtMatch = filesGT(strcmp(gtNames, ID_GT));
    
    if isempty(gtMatch)
        warning('No ground truth found for %s', segName);
        continue
    end
    
    Igt = imread(fullfile(folderGT, gtMatch(1).name));
    if ~islogical(Igt)
        Igt = Igt > 0;
    end
    
    if ~isequal(size(Iseg), size(Igt))
        Iseg = imresize(Iseg, [size(Igt,1), size(Igt,2)], 'nearest');
    end
    
    intersection = sum(Iseg(:) & Igt(:));
    union = sum(Iseg(:) | Igt(:));
    
    jaccard = intersection / union;
    dice = 2 * intersection / (sum(Iseg(:)) + sum(Igt(:)));
    
    results.Image(k) = segName;
    results.Method(k) = method;
    results.Jaccard(k) = jaccard;
    results.Dice(k) = dice;
end

% --- Save results ---
outputFile = fullfile('C:\Users\Usuario\Desktop', 'Results_Jaccard_Dice_2.xlsx');
writetable(results, outputFile);

% --- Calculate average per method ---
methods = unique(results.Method);
averages = table('Size',[length(methods),3], ...
                  'VariableTypes',{'string','double','double'}, ...
                  'VariableNames',{'Method','Jaccard_Mean','Dice_Mean'});

for i = 1:length(methods)
    idx = results.Method == methods(i);
    averages.Method(i) = methods(i);
    averages.Jaccard_Mean(i) = mean(results.Jaccard(idx));
    averages.Dice_Mean(i) = mean(results.Dice(idx));
end

outputFileProm = fullfile('C:\Users\Usuario\Desktop', 'Averages_Jaccard_Dice_2.xlsx');
writetable(averages, outputFileProm);

disp('--- Average per method ---');
disp(averages);

% --- Bar graph ---
figure('Name','Method Comparison','Color','w');
barData = [averages.Jaccard_Mean, averages.Dice_Mean];
b = bar(categorical(averages.Method), barData);
b(1).FaceColor = [0.2 0.6 0.8]; % Jaccard color
b(2).FaceColor = [0.9 0.4 0.4]; % Dice color
legend({'Jaccard','Dice'}, 'Location','northoutside','Orientation','horizontal');
ylabel('Value');
title('Average Jaccard and Dice by method');
xtickangle(45);
grid on;
