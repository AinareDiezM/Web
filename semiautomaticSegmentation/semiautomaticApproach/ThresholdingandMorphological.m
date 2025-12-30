%% ====== ROI image segmentation ======
clc; clear;

% Folder containing the ROI images
folder_path = 'C:\Users\Usuario\Desktop\BIALYSTOK\SEGMENTACIONES\FINAL\Seleccion ADC y SCC png\Imagenesfinales';

% Folder to save segmentations
outputFolder = fullfile(folder_path,'Thresholding segmentation');
if ~exist(outputFolder,'dir')
    mkdir(outputFolder);
end

% Valid extensions
exts = {'*.png','*.jpg','*.tif','*.dcm'};

% List files
files = [];
for k = 1:length(exts)
    files = [files; dir(fullfile(folder_path,'**',exts{k}))];
end

%% Segmentation parameters
level_global = 0.265; % mean Otsu
sensitivity_adapt = 0.5; % sensitivity for adaptive thresholding

%% Process each image
for i = 1:length(files)
    file_path = fullfile(files(i).folder, files(i).name);
    [~, name, ~] = fileparts(files(i).name);
    
    % Read image
    [~,~,ext] = fileparts(file_path);
    if strcmpi(ext,'.dcm')
        img = double(dicomread(file_path));
    else
        img = double(imread(file_path));
    end
    
    % Normalize to [0, 1]
    img_norm = (img - min(img(:))) / (max(img(:)) - min(img(:)));
    
    %% ====== Thresholding ======
    bw_global = imbinarize(img_norm, level_global); % Global threshold
    bw_adapt  = imbinarize(img_norm, 'adaptive', 'Sensitivity', sensitivity_adapt); % Adaptive threshold
    level_otsu = graythresh(img_norm);
    bw_otsu = imbinarize(img_norm, level_otsu); % Per-image Otsu
     %% ====== Morphological operators ======
    se = strel('disk', 3);
    
    % Apply morphological operations based on the article
    bw_eroded  = imerode(bw_otsu,  se);    % reduces region, cleans thin edges
    bw_dilated = imdilate(bw_otsu, se);    % expands region, fills small holes
    bw_open   = imopen(bw_otsu,   se);     % opening to remove small noise
    bw_close  = imclose(bw_otsu,  se);     % closing to fill gaps
    
    % Save results for comparison
    imwrite(bw_eroded,  fullfile(outputFolder, [name '_erode.png']));
    imwrite(bw_dilated, fullfile(outputFolder, [name '_dilate.png']));
    imwrite(bw_open,    fullfile(outputFolder, [name '_open.png']));
    imwrite(bw_close,   fullfile(outputFolder, [name '_close.png']));
    %% ====== Save results ======
    imwrite(bw_global, fullfile(outputFolder, [name '_global.png']));
    imwrite(bw_adapt,  fullfile(outputFolder, [name '_adapt.png']));
    imwrite(bw_otsu,   fullfile(outputFolder, [name '_otsu.png']));

    fprintf('Segmented: %s\n', files(i).name);
end

disp('All images have been segmented using the 3 methods.');
