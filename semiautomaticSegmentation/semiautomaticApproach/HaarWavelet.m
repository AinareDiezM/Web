%% ====== MRI Segmentation: Haar Wavelet ======
clc; clear;

folder_path = 'C:\Users\Usuario\Desktop\BIALYSTOK\SEGMENTACIONES\FINAL\Seleccion ADC y SCC png\Imagenesfinales';
outputFolder = fullfile(folder_path,'Wavelet_Haar_Segmentation');
if ~exist(outputFolder,'dir')
    mkdir(outputFolder);
end

exts = {'*.png'};
files = [];
for k = 1:length(exts)
    files = [files; dir(fullfile(folder_path,'**',exts{k}))];
end

fprintf('Processing %d images...\n', numel(files));

for i = 1:length(files)
    file_path = fullfile(files(i).folder, files(i).name);
    [~, name, ext] = fileparts(files(i).name);
    
    if strcmpi(ext,'.dcm')
        img = double(dicomread(file_path));
    else
        img = double(imread(file_path));
    end
    
    img_norm = (img - min(img(:))) / (max(img(:)) - min(img(:)));

    %% Haar wavelet
    [cA, ~, ~, ~] = dwt2(img_norm, 'haar');
    bw_wavelet = imbinarize(cA);

    %% Resize to 256x256
    bw_wavelet_resized = imresize(bw_wavelet, [256 256]);

    %% Convert to uint8 to avoid saving errors
    bw_wavelet_resized_uint8 = uint8(bw_wavelet_resized * 255);

    %% Save
    imwrite(bw_wavelet_resized_uint8, fullfile(outputFolder, [name '_wavelet.png']));

    fprintf('Segmented (Wavelet) and resized: %s\n', files(i).name);
end

disp('--- All images have been processed with Haar Wavelet and are 256x256 ---');
