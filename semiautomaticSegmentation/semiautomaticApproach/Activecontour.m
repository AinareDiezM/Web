%% === Seed-based Active Contours ===
clear; clc; close all;

%% === Folder configuration ===
input_folder = 'C:\Users\Usuario\Desktop\BIALYSTOK\SEGMENTACIONES\FINAL\Seleccion ADC y SCC png\Imagenesfinales';
output_folder = fullfile(input_folder, 'Segmentacion_ActiveContour_Seeds');
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

%% === Manually define images and seeds ===
% Format: {'image_name', [x_seed, y_seed]}
seeds_manual = {
    'ADC_p1_ROI.png', [143, 119];
    'ADC_p2_ROI.png', [127, 127];
    'ADC_p3_ROI.png', [126, 128];
    'ADC_p4_ROI.png', [131, 129];
    'ADC_p5_ROI.png', [123, 132];
    'SCC_p1_ROI.png', [124, 131];
    'SCC_p2_ROI.png', [127, 129];
    'SCC_p3_ROI.png', [121, 134];
    'SCC_p4_ROI.png', [129, 131];
    'SCC_p5_ROI.png', [134, 129];
   
};

%% === Active Contour method parameters ===
radius = 50;          % initial radius around the seed
num_iter = 300;       % number of iterations
min_obj_size = 50;    % minimum object size to keep
sigma_smooth = 1;     % Gaussian smoothing before segmentation

%% === Process each image ===
for i = 1:size(seeds_manual,1)
    name = seeds_manual{i,1};
    seed_x = seeds_manual{i,2}(1);
    seed_y = seeds_manual{i,2}(2);

    img_path = fullfile(input_folder, name);
    if ~isfile(img_path)
        warning('Image not found: %s', img_path);
        continue;
    end

    % Read image and normalize
    img = im2double(imread(img_path));
    if size(img,3) > 1
        img = rgb2gray(img);
    end
    img_norm = mat2gray(img);

    % Preprocessing: adaptive contrast enhancement and smoothing
    img_norm = adapthisteq(img_norm);
    img_norm = imgaussfilt(img_norm, sigma_smooth);

    % Create initial circular mask centered at the seed
    [xx, yy] = meshgrid(1:size(img_norm,2), 1:size(img_norm,1));
    init_mask = ( (xx - seed_x).^2 + (yy - seed_y).^2 ) <= radius^2;

    % Apply region-based Active Contour (Chan–Vese)
    bw_snake = activecontour(img_norm, init_mask, num_iter, 'Chan-Vese');

    % Post-processing: clean and fill regions
    bw_snake = imfill(bw_snake, 'holes');
    bw_snake = bwareaopen(bw_snake, min_obj_size);

    % Save result
    [~, base, ~] = fileparts(name);
    out_path = fullfile(output_folder, [base '_activecontour.png']);
    imwrite(bw_snake, out_path);

    % Optional visualization
    figure(1); clf;
    imshow(img_norm, []); hold on;
    visboundaries(bw_snake, 'Color', 'r', 'LineWidth', 1);
    plot(seed_x, seed_y, 'yo', 'MarkerFaceColor', 'y');
    title(sprintf('Active Contour Chan–Vese - %s', name));
    drawnow

    fprintf('Segmented: %s\n', name);
end

disp('--- Active Contour segmentation completed ---');
