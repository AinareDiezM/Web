% ==========================================
% VALIDATION STUDY 2 – MANUAL vs AUTOMATIC
% Patient-level metrics + plots + error maps
% ==========================================
clear; clc;

% --- FOLDERS ---
baseGT   = 'C:\Users\Usuario\Desktop\VALIDATION\GT_RAS_PNG_RECORTE';   % Manual masks (ground truth)
baseAUTO = 'C:\Users\Usuario\Desktop\VALIDATION\AUTO - ROI';           % Automatic masks (U-Net / Model 9, etc.)

% List all PNG files in GT folder
gtFiles = dir(fullfile(baseGT, '*.png'));

% Cell array to store patient-level results
resultsPatient = {};

for i = 1:length(gtFiles)
    
    gtName = gtFiles(i).name;         
    [~, patientID, ~] = fileparts(gtName);  
    
    % Corresponding automatic mask 
    autoPath = fullfile(baseAUTO, gtName);
    
    if ~exist(autoPath, 'file')
        fprintf('Automatic mask missing for patient: %s\n', patientID);
        continue;
    end
    
    % ------------------------------------------
    % Read masks
    % ------------------------------------------
    gtMask   = imread(fullfile(baseGT, gtName));
    autoMask = imread(autoPath);
    
    % Convert to binary masks
    gtBin   = gtMask > 0;
    autoBin = autoMask > 0;
    
    % Ensure same size
    if ~isequal(size(gtBin), size(autoBin))
        warning('Size mismatch GT vs AUTO for patient %s', patientID);
        continue;
    end
    
    % ------------------------------------------
    % Compute confusion components
    % ------------------------------------------
    TP = sum(gtBin(:) & autoBin(:));
    FP = sum(~gtBin(:) & autoBin(:));
    FN = sum(gtBin(:) & ~autoBin(:));
    TN = sum(~gtBin(:) & ~autoBin(:));
    
    % ------------------------------------------
    % Compute metrics
    % ------------------------------------------
    dice = (2 * TP) / (2 * TP + FP + FN + eps);
    iou  = TP / (TP + FP + FN + eps);
    prec = TP / (TP + FP + eps);
    rec  = TP / (TP + FN + eps);
    
    % F1-score and Accuracy
    f1  = (2 * prec * rec) / (prec + rec + eps);
    acc = (TP + TN) / (TP + TN + FP + FN + eps);
    
    % ------------------------------------------
    % Tumour volume in pixels
    % ------------------------------------------
    volGT   = sum(gtBin(:));
    volAuto = sum(autoBin(:));
    
    volErrorAbs = volAuto - volGT;
    volErrorPct = 100 * volErrorAbs / (volGT + eps);
    
    % ------------------------------------------
    % Store results for this patient
    % (also store masks for later error maps)
    % ------------------------------------------
    resultsPatient = [resultsPatient; ...
        {patientID, dice, iou, prec, rec, f1, acc, ...
         volGT, volAuto, volErrorAbs, volErrorPct, ...
         gtBin, autoBin}];
end

% ------------------------------------------
% Convert to table and save to CSV
% ------------------------------------------
resultsPatientTable = cell2table(resultsPatient, ...
    'VariableNames', {'PatientID','Dice','IoU','Precision','Recall','F1','Accuracy', ...
                      'TotalVolGT','TotalVolAuto','VolErrorAbs','VolErrorPct', ...
                      'GTmask','AUTOmask'});

outPath = 'C:\Users\Usuario\Desktop\VALIDATION\RESULTS_Val2_Patient.csv';
writetable(resultsPatientTable, outPath);
fprintf('Validation Study 2 – patient-level results saved to:\n%s\n', outPath);


% ==========================================
% PART 2: SCATTER PLOT (Manual vs Automatic Volumes)
% ==========================================

figure;
scatter(resultsPatientTable.TotalVolGT, resultsPatientTable.TotalVolAuto, 80, 'filled');
hold on;

minV = min([resultsPatientTable.TotalVolGT; resultsPatientTable.TotalVolAuto]);
maxV = max([resultsPatientTable.TotalVolGT; resultsPatientTable.TotalVolAuto]);
plot([minV maxV], [minV maxV], 'k--', 'LineWidth', 1.2);

xlabel('Manual Volume (pixels)');
ylabel('Automatic Volume (pixels)');
title('Validation Study 2 – Manual vs Automatic Tumour Volume');
grid on;

% Annotate each point with patient ID
for i = 1:height(resultsPatientTable)
    text(resultsPatientTable.TotalVolGT(i), ...
         resultsPatientTable.TotalVolAuto(i), ...
         ['  ' resultsPatientTable.PatientID{i}], ...
         'FontSize', 8);
end

% Remove axes toolbar for clean export
set(gcf, 'Toolbar', 'none');
set(gcf, 'MenuBar', 'none');

saveas(gcf, 'C:\Users\Usuario\Desktop\VALIDATION\FIG_Val2_Scatter_Volumes.png');

% ==========================================
% PART 3: ERROR MAPS FOR MANUALLY SELECTED CASES
% ==========================================

% Manually define best and worst patients 
bestPatientID  = 'SCC_P4';   % best-performing case
worstPatientID = 'ADC_P3';   % worst-performing case 

% Find indices in the table
idxBest  = strcmp(resultsPatientTable.PatientID, bestPatientID);
idxWorst = strcmp(resultsPatientTable.PatientID, worstPatientID);

if ~any(idxBest)
    error('Best patient ID "%s" not found in resultsPatientTable.', bestPatientID);
end
if ~any(idxWorst)
    error('Worst patient ID "%s" not found in resultsPatientTable.', worstPatientID);
end

% Print confirmation in the command window
fprintf('Selected BEST case:  %s (Dice = %.3f)\n', ...
    bestPatientID, resultsPatientTable.Dice(idxBest));
fprintf('Selected WORST case: %s (Dice = %.3f)\n', ...
    worstPatientID, resultsPatientTable.Dice(idxWorst));


% ---------- WORST CASE ERROR MAP ----------
GT   = resultsPatientTable.GTmask{idxWorst};
AUTO = resultsPatientTable.AUTOmask{idxWorst};

TPmap = GT & AUTO;
FPmap = ~GT & AUTO;
FNmap = GT & ~AUTO;

errorRGB = zeros(size(GT,1), size(GT,2), 3);
errorRGB(:,:,2) = TPmap;   % green
errorRGB(:,:,1) = FPmap;   % red
errorRGB(:,:,3) = FNmap;   % blue;

figure;
imshow(errorRGB);
title(['Validation Study 2 – Error Map (Worst case: ', worstPatientID, ')']);
hold on;

% Legend using invisible markers
hTP = plot(nan, nan, 's', 'MarkerFaceColor', [0 1 0], 'MarkerEdgeColor', [0 1 0], 'MarkerSize', 12);
hFP = plot(nan, nan, 's', 'MarkerFaceColor', [1 0 0], 'MarkerEdgeColor', [1 0 0], 'MarkerSize', 12);
hFN = plot(nan, nan, 's', 'MarkerFaceColor', [0 0 1], 'MarkerEdgeColor', [0 0 1], 'MarkerSize', 12);

legend([hTP hFP hFN], ...
    {'True Positive (TP)', 'False Positive (FP)', 'False Negative (FN)'}, ...
    'Location', 'southoutside', ...
    'Orientation', 'horizontal', ...
    'FontSize', 10, ...
    'Box', 'off');

set(gcf, 'Toolbar', 'none');
set(gcf, 'MenuBar', 'none');
saveas(gcf, ['C:\Users\Usuario\Desktop\VALIDATION\FIG_Val2_ErrorMap_Worst_' worstPatientID '.png']);


% ---------- BEST CASE ERROR MAP ----------
GT   = resultsPatientTable.GTmask{idxBest};
AUTO = resultsPatientTable.AUTOmask{idxBest};

TPmap = GT & AUTO;
FPmap = ~GT & AUTO;
FNmap = GT & ~AUTO;

errorRGB = zeros(size(GT,1), size(GT,2), 3);
errorRGB(:,:,2) = TPmap;   % green
errorRGB(:,:,1) = FPmap;   % red
errorRGB(:,:,3) = FNmap;   % blue;

figure;
imshow(errorRGB);
title(['Validation Study 2 – Error Map (Best case: ', bestPatientID, ')']);
hold on;

hTP = plot(nan, nan, 's', 'MarkerFaceColor', [0 1 0], 'MarkerEdgeColor', [0 1 0], 'MarkerSize', 12);
hFP = plot(nan, nan, 's', 'MarkerFaceColor', [1 0 0], 'MarkerEdgeColor', [1 0 0], 'MarkerSize', 12);
hFN = plot(nan, nan, 's', 'MarkerFaceColor', [0 0 1], 'MarkerEdgeColor', [0 0 1], 'MarkerSize', 12);

legend([hTP hFP hFN], ...
    {'True Positive (TP)', 'False Positive (FP)', 'False Negative (FN)'}, ...
    'Location', 'southoutside', ...
    'Orientation', 'horizontal', ...
    'FontSize', 10, ...
    'Box', 'off');

axtoolbar(gca, 'none');
saveas(gcf, ['C:\Users\Usuario\Desktop\VALIDATION\FIG_Val2_ErrorMap_Best_' bestPatientID '.png']);
