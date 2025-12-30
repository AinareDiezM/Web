
%% Parameters
inputFolder = 'C:\Users\Usuario\Desktop\DEMO\PRUEBAAUTO';
outputFolder = 'C:\Users\Usuario\Desktop\DEMO\PRUEBAAUTO - ROI';
%% Parameters
outputSize = [256 256];  

if ~exist(outputFolder,'dir')
    mkdir(outputFolder);
end

imgFiles = dir(fullfile(inputFolder,'*.png'));
roi_info = struct; % Store ROI coordinates per patient

for i = 1:length(imgFiles)
    imgPath = fullfile(inputFolder,imgFiles(i).name);
    img = imread(imgPath);
    [~, name, ext] = fileparts(imgFiles(i).name);

    % --- Extract patient ID (first two tokens)
    tokens = split(name, '_');
    if length(tokens) < 2
        warning('Cannot determine patient ID in: %s', name);
        continue;
    end
    patientID = strjoin(tokens(1:2), '_');  

    % --- Define ROI only once per patient
    if ~isfield(roi_info, patientID)
        hFig = figure('Name', sprintf('Define ROI: %s', name));
        imshow(img, []);
        title('Draw a rectangle over the tumour and press Enter');

        h = imrect;              % Interactive rectangular ROI
        pos = wait(h);           % Returns [x, y, width, height]
        close(hFig);

        % Save coordinates
        roi_info.(patientID).coords = round(pos);  % [x, y, width, height]
    end

    % --- Apply ROI to the image
    coords = roi_info.(patientID).coords;
    x1 = coords(1);
    y1 = coords(2);
    x2 = min(x1 + coords(3), size(img,2));
    y2 = min(y1 + coords(4), size(img,1));

    cropped_img = img(y1:y2, x1:x2, :);

    % Optionally resize
    resized_img = imresize(cropped_img, outputSize);

    % --- Save cropped image
    outName = fullfile(outputFolder, [name '' ext]);
    imwrite(resized_img, outName);

    fprintf('Image %s cropped using ROI from %s\n', name, patientID);
end

%% Save coordinates for all patients
save(fullfile(outputFolder,'roi_info.mat'),'roi_info');
disp('All images processed and coordinates saved.');
