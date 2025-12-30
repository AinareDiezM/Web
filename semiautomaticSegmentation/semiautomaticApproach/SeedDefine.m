%% ====== Seed selection in ALL PNG images ======
clear; clc; close all;

% Folder containing the PNG images
img_folder = 'C:\Users\Usuario\Desktop\BIALYSTOK\SEGMENTACIONES\FINAL\Seleccion ADC y SCC png\Imagenesfinales';

% List PNG files
png_files_struct = dir(fullfile(img_folder, '*.png'));
if isempty(png_files_struct)
    error('No PNG images were found in the specified folder.');
end

% Sort alphabetically by file name
png_files = {png_files_struct.name};
png_files = sort(png_files);

% Table to store seed coordinates
seed_data = table('Size', [0 3], ...
    'VariableTypes', {'string','double','double'}, ...
    'VariableNames', {'Image','X','Y'});

fprintf('A total of %d images were found. Please select one seed per image.\n', numel(png_files));

for i = 1:numel(png_files)
    fname = png_files{i};
    img_path = fullfile(img_folder, fname);
    img = imread(img_path);
    
    figure; imshow(img, []);
    title(sprintf('Select the seed for: %s', fname));
    
    % Manual selection
    [x, y] = ginput(1);
    x = round(x); y = round(y);
    close;
    
    % Store in table
    seed_data = [seed_data; {fname, x, y}];
end

% Display results
disp('Selected seeds:');
disp(seed_data);

% Save to CSV
output_csv = fullfile(img_folder, 'all_seeds.csv');
writetable(seed_data, output_csv);
fprintf('Coordinates saved to: %s\n', output_csv);
