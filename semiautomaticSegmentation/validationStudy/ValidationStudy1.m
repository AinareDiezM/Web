% ==========================================
%VALIDATION STUDY 2 – MANUAL vs SEMI-AUTOMATIC
% Slice-level metrics
% ==========================================
clear; clc;

% --- FOLDERS ---
baseGT   = 'C:\Users\Usuario\Desktop\VALIDATION\GT_RAS_PNG_RECORTE';        % Manual masks (ground truth)
baseSEMI = 'C:\Users\Usuario\Desktop\VALIDATION\MEJOR SEMIAUTOMATICO';      % Semi-automatic masks

% ------------------------------------------
% PART 1: Load file names and compute metrics
% ------------------------------------------

gtFiles = dir(fullfile(baseGT, '*.png'));
resultsPatient = {};

for i = 1:length(gtFiles)
    
    gtName = gtFiles(i).name;     
    [~, patientID, ~] = fileparts(gtName); 
    
    % Path to semi-automatic mask
    semiPath = fullfile(baseSEMI, gtName);
    
    if ~exist(semiPath, 'file')
        fprintf('Semi-automatic mask missing: %s\n', gtName);
        continue;
    end
    
    % --- Read masks ---
    gtMask   = imread(fullfile(baseGT, gtName));
    semiMask = imread(semiPath);
    
    % Convert to binary masks
    gtBin   = gtMask > 0;
    semiBin = semiMask > 0;
    
    % Ensure same size
    if ~isequal(size(gtBin), size(semiBin))
        warning('Size mismatch for patient %s', patientID);
        continue;
    end
    
    % --- Compute confusion components ---
    TP = sum(gtBin(:) & semiBin(:));
    FP = sum(~gtBin(:) & semiBin(:));
    FN = sum(gtBin(:) & ~semiBin(:));
    TN = sum(~gtBin(:) & ~semiBin(:)); 

    % --- Metrics ---
    dice = (2*TP) / (2*TP + FP + FN + eps);
    iou  = TP / (TP + FP + FN + eps);
    prec = TP / (TP + FP + eps);
    rec  = TP / (TP + FN + eps);
    
    % --- Volume in pixels ---
    volGT   = sum(gtBin(:));
    volSemi = sum(semiBin(:));
    volErrorAbs = volSemi - volGT;
    volErrorPct = 100 * volErrorAbs / (volGT + eps);
    
    % Store results
    resultsPatient = [resultsPatient; ...
        {patientID, dice, iou, prec, rec, ...
         volGT, volSemi, volErrorAbs, volErrorPct, ...
         gtBin, semiBin}];   % <-- ALSO store masks for error maps
end

% Convert to table
resultsPatientTable = cell2table(resultsPatient, ...
    'VariableNames', {'PatientID','Dice','IoU','Precision','Recall', ...
                      'TotalVolGT','TotalVolSemi','VolErrorAbs','VolErrorPct', ...
                      'GTmask','SEMIpmask'});

% Save CSV
outPath = 'C:\Users\Usuario\Desktop\DEMO\RESULTS_Exp1_Patient.csv';
writetable(resultsPatientTable, outPath);
fprintf('Patient-level results saved to:\n%s\n', outPath);


% ==========================================
% PART 2: SCATTER PLOT (Manual vs Semi-Automatic Volumes)
% ==========================================

figure;
scatter(resultsPatientTable.TotalVolGT, resultsPatientTable.TotalVolSemi, ...
        80, 'filled');  % point size 80
hold on;

% Plot identity line (perfect agreement)
minV = min([resultsPatientTable.TotalVolGT; resultsPatientTable.TotalVolSemi]);
maxV = max([resultsPatientTable.TotalVolGT; resultsPatientTable.TotalVolSemi]);
plot([minV maxV], [minV maxV], 'k--', 'LineWidth', 1.2);

xlabel('Manual Volume (pixels)');
ylabel('Semi-Automatic Volume (pixels)');
title('Validation Study 1 – Manual vs Semi-Automatic Tumour Volume');
grid on;

% Annotate each point with patient ID
for i = 1:height(resultsPatientTable)
    text(resultsPatientTable.TotalVolGT(i), ...
         resultsPatientTable.TotalVolSemi(i), ...
         ['  ' resultsPatientTable.PatientID{i}], ...
         'FontSize', 8);
end

% Save figure
saveas(gcf, 'C:\Users\Usuario\Desktop\VALIDATION\FIG_Exp1_Scatter_Volumes.png');

% ==========================================
% PART 3: ERROR MAPS WITH LEGEND
% ==========================================

selectedPatient = 'SCC_P2';  

idx = strcmp(resultsPatientTable.PatientID, selectedPatient);
if any(idx)

    GT   = resultsPatientTable.GTmask{idx};
    SEMI = resultsPatientTable.SEMIpmask{idx};

    TPmap = GT & SEMI;
    FPmap = ~GT & SEMI;
    FNmap = GT & ~SEMI;

    errorRGB = zeros(size(GT,1), size(GT,2), 3);
    errorRGB(:,:,2) = TPmap;   % green
    errorRGB(:,:,1) = FPmap;   % red
    errorRGB(:,:,3) = FNmap;   % blue

    figure;
    imshow(errorRGB);
    title(['Validation Study 1 – Error Map (', selectedPatient, ')']);

    hold on;

    % ------- REAL LEGEND (using invisible plot handles) -------
    hTP = plot(nan, nan, 's', 'MarkerFaceColor', [0 1 0], 'MarkerEdgeColor', [0 1 0], 'MarkerSize', 12);
    hFP = plot(nan, nan, 's', 'MarkerFaceColor', [1 0 0], 'MarkerEdgeColor', [1 0 0], 'MarkerSize', 12);
    hFN = plot(nan, nan, 's', 'MarkerFaceColor', [0 0 1], 'MarkerEdgeColor', [0 0 1], 'MarkerSize', 12);

    legend([hTP hFP hFN], ...
        {'True Positive (TP)', 'False Positive (FP)', 'False Negative (FN)'}, ...
        'Location', 'southoutside', ...
        'Orientation', 'horizontal', ...
        'FontSize', 12, ...
        'Box', 'off');

    % Remove toolbar
    axtoolbar(gca, 'none');

    % Save figure
    saveas(gcf, ['C:\Users\Usuario\Desktop\VALIDATION\FIG_Exp1_ErrorMap_' selectedPatient '.png']);

else
    warning('Patient %s not found.', selectedPatient);
end
