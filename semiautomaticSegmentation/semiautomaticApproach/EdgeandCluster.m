clear; clc; close all;

%% === Configuration ===
input_folder = 'C:\Users\Usuario\Desktop\BIALYSTOK\SEGMENTACIONES\FINAL\Seleccion ADC y SCC png\Imagenesfinales';
output_folder = 'C:\Users\Usuario\Desktop\BIALYSTOK\SEGMENTACIONES\FINAL\Seleccion ADC y SCC png\EXTRAS';

methods = {'canny','sobel','prewitt'}; % Edge detection methods

if ~exist(output_folder,'dir')
    mkdir(output_folder);
end

%% === List PNG images directly from the folder ===
files = dir(fullfile(input_folder,'*.png'));  
if isempty(files)
    warning('No PNG images were found in %s', input_folder);
end

for f = 1:length(files)
    file_path = fullfile(input_folder, files(f).name);
    img = im2double(imread(file_path));  % Read PNG image as double [0,1]

    % --- Normalization and contrast adjustment ---
    img_norm = (img - min(img(:))) / (max(img(:)) - min(img(:)));
    img_norm = imadjust(img_norm);

    [~, name, ~] = fileparts(files(f).name);

    %% ====== 1. Edge Detection ======
    for m = 1:length(methods)
        method = methods{m};
        switch method
            case 'canny'
                bw = edge(img_norm,'Canny',[0.1 0.3],1.0);
            case 'sobel'
                bw = edge(img_norm,'Sobel');
            case 'prewitt'
                bw = edge(img_norm,'Prewitt');
            otherwise
                bw = false(size(img_norm));
        end

        out_name = sprintf('%s_%s.png',name,method);
        out_path = fullfile(output_folder,out_name);
        imwrite(bw,out_path);
    end

    %% ====== 2. K-means Clustering ======
    nClusters = 2;
    [L, ~] = imsegkmeans(single(img_norm), nClusters);

    clusterMeans = zeros(1,nClusters);
    for k = 1:nClusters
        clusterMeans(k) = mean(img_norm(L==k));
    end
    [~, tumorCluster] = max(clusterMeans);

    bw_kmeans = L == tumorCluster;

    % Save mask
    out_name = sprintf('%s_kmeans.png',name);
    imwrite(bw_kmeans, fullfile(output_folder,out_name));

    %% ====== 3. Fuzzy C-Means Clustering ======
    try
        data = img_norm(:);
        [centers,U] = fcm(data, nClusters, [2.0, 100, 1e-5, 0]);

        % Select cluster with the highest centroid value (tumour)
        [~, tumorCluster] = max(centers);
        [~, maxU] = max(U);  % each pixel is assigned to the cluster with highest membership
        bw_fcm = reshape(maxU-1 == (tumorCluster-1), size(img_norm));

        % Save mask
        out_name = sprintf('%s_fcm.png',name);
        imwrite(bw_fcm, fullfile(output_folder,out_name));
    catch
        warning('Fuzzy C-Means not available for %s', files(f).name);
    end

end
