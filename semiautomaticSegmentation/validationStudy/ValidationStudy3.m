% ==========================================
% VOLUMETRIC TUMOUR SIZE CONSISTENCY
% ==========================================
clear; clc;

% ------------------------------------------
% 1. LOAD RESULTS FROM VALIDATION STUDY 1 & 2
% ------------------------------------------
% Paths to CSV files 
exp1Path = 'C:\Users\Usuario\Desktop\DEMO\RESULTS_Exp1_Patient.csv';        % Semi-automatic
val2Path = 'C:\Users\Usuario\Desktop\VALIDATION\RESULTS_Val2_Patient.csv';  % Automatic

Tsemi = readtable(exp1Path);
Tauto = readtable(val2Path);

% Keep only relevant columns and rename to avoid conflicts
TsemiSub = Tsemi(:, {'PatientID','TotalVolGT','TotalVolSemi','VolErrorPct'});
TautoSub = Tauto(:, {'PatientID','TotalVolGT','TotalVolAuto','VolErrorPct'});

TsemiSub.Properties.VariableNames = {'PatientID','ManualVol_semi','SemiVol','SemiErrPct'};
TautoSub.Properties.VariableNames = {'PatientID','ManualVol_auto','AutoVol','AutoErrPct'};

% Join both tables by PatientID (inner join to keep common patients)
Tall = innerjoin(TsemiSub, TautoSub, 'Keys', 'PatientID');

% For safety, choose one manual volume reference (they should match)
ManualVol = Tall.ManualVol_semi;   % or Tall.ManualVol_auto, if identical
SemiVol   = Tall.SemiVol;
AutoVol   = Tall.AutoVol;

SemiErrPct = Tall.SemiErrPct;
AutoErrPct = Tall.AutoErrPct;

patients = Tall.PatientID;

% ==========================================
% FIGURE A – SCATTER PLOT (Manual vs Semi vs Automatic)
% ==========================================
figure;
hold on;

% Scatter for semi-automatic (blue)
scatter(ManualVol, SemiVol, 80, 'filled', 'MarkerFaceColor', [0 0.4470 0.7410]);

% Scatter for automatic (orange)
scatter(ManualVol, AutoVol, 80, 'filled', 'MarkerFaceColor', [0.8500 0.3250 0.0980]);

% Identity line (black, perfect agreement)
minV = min([ManualVol; SemiVol; AutoVol]);
maxV = max([ManualVol; SemiVol; AutoVol]);
plot([minV maxV], [minV maxV], 'k--', 'LineWidth', 1.2);

% Labels and title
xlabel('Manual Volume (pixels)');
ylabel('Estimated Volume (pixels)');
title('Volumetric Consistency – Manual vs Semi-Automatic vs Automatic');
grid on;

% Legend
legend({'Semi-automatic','Automatic','Identity line'}, 'Location', 'best');


for i = 1:numel(patients)
    text(ManualVol(i), SemiVol(i), ['  ' patients{i}], 'FontSize', 8, 'Color', [0 0.4470 0.7410]);
    text(ManualVol(i), AutoVol(i), ['  ' patients{i}], 'FontSize', 8, 'Color', [0.8500 0.3250 0.0980]);
end

% Save figure
saveas(gcf, 'C:\Users\Usuario\Desktop\VALIDATION\FIG_Volume_Scatter_Manual_Semi_Auto.png');


% ==========================================
% FIGURE B – BLAND–ALTMAN PLOTS
% (Manual vs Semi, Manual vs Automatic)
% ==========================================

% --- Manual vs Semi-automatic ---
meanSemi = (ManualVol + SemiVol) / 2;
diffSemi = SemiVol - ManualVol;   % difference = method - reference

meanDiffSemi = mean(diffSemi);
sdDiffSemi   = std(diffSemi);
LoA_low_Semi  = meanDiffSemi - 1.96*sdDiffSemi;
LoA_high_Semi = meanDiffSemi + 1.96*sdDiffSemi;

% --- Manual vs Automatic ---
meanAuto = (ManualVol + AutoVol) / 2;
diffAuto = AutoVol - ManualVol;

meanDiffAuto = mean(diffAuto);
sdDiffAuto   = std(diffAuto);
LoA_low_Auto  = meanDiffAuto - 1.96*sdDiffAuto;
LoA_high_Auto = meanDiffAuto + 1.96*sdDiffAuto;

figure;

% ---------- Subplot 1: Manual vs Semi ----------
subplot(1,2,1);
scatter(meanSemi, diffSemi, 60, 'filled', 'MarkerFaceColor', [0 0.4470 0.7410]);
hold on;
yline(meanDiffSemi, 'k-', 'LineWidth', 1.2);
yline(LoA_low_Semi, 'k--', 'LineWidth', 1.0);
yline(LoA_high_Semi,'k--', 'LineWidth', 1.0);
xlabel('Mean volume (pixels)');
ylabel('Semi - Manual (pixels)');
title('Bland–Altman: Manual vs Semi-automatic');
grid on;

% ---------- Subplot 2: Manual vs Automatic ----------
subplot(1,2,2);
scatter(meanAuto, diffAuto, 60, 'filled', 'MarkerFaceColor', [0.8500 0.3250 0.0980]);
hold on;
yline(meanDiffAuto, 'k-', 'LineWidth', 1.2);
yline(LoA_low_Auto, 'k--', 'LineWidth', 1.0);
yline(LoA_high_Auto,'k--', 'LineWidth', 1.0);
xlabel('Mean volume (pixels)');
ylabel('Automatic - Manual (pixels)');
title('Bland–Altman: Manual vs Automatic');
grid on;

% Save figure
saveas(gcf, 'C:\Users\Usuario\Desktop\VALIDATION\FIG_Volume_BlandAltman_Semi_Auto.png');


% ==========================================
% FIGURE C – BAR PLOT OF VOLUME ERROR (%)
% ==========================================

% Build matrix: rows = patients, columns = [Semi, Auto]
errMatrix = [SemiErrPct, AutoErrPct];

figure;
bar(errMatrix, 'grouped');
xlabel('Patient');
ylabel('Volume Error (%)');
title('Relative Volume Error – Semi-automatic vs Automatic');
grid on;

% X-axis labels = patient IDs
set(gca, 'XTick', 1:numel(patients), 'XTickLabel', patients);
xtickangle(45);  % rotate labels for readability

legend({'Semi-automatic','Automatic'}, 'Location', 'best');

% Save figure
saveas(gcf, 'C:\Users\Usuario\Desktop\VALIDATION\FIG_Volume_Error_Barplot.png');
