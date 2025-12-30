% ================================================
% COMPUTE ADC vs SCC PERFORMANCE STATISTICS
% ================================================
clear; clc;

% Load results table
T = readtable('C:\Users\Usuario\Desktop\VALIDATION\RESULTS_Val2_Patient.csv');

% Identify subtype from PatientID
isADC = contains(T.PatientID, "ADC", 'IgnoreCase', true);
isSCC = contains(T.PatientID, "SCC", 'IgnoreCase', true);

% Subset tables
T_ADC = T(isADC, :);
T_SCC = T(isSCC, :);

% Metrics to analyse
metrics = {'Dice','IoU','Precision','Recall','F1','Accuracy','VolErrorPct'};

% Prepare result matrix
subtypeStats = table;
subtypeStats.Subtype = ["ADC"; "SCC"];

for m = 1:length(metrics)
    metric = metrics{m};

    ADC_mean = mean(T_ADC.(metric));
    ADC_std  = std(T_ADC.(metric));

    SCC_mean = mean(T_SCC.(metric));
    SCC_std  = std(T_SCC.(metric));

    subtypeStats.([metric '_Mean']) = [ADC_mean; SCC_mean];
    subtypeStats.([metric '_Std'])  = [ADC_std; SCC_std];
end

% Display table
disp(subtypeStats);

% Save table
writetable(subtypeStats, 'C:\Users\Usuario\Desktop\VALIDATION\ADC_SCC_Stats.csv');
fprintf('Saved subtype statistics to ADC_SCC_Stats.csv\n');
