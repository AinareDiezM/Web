%% ==== CONFIGURATION ==========================================
rootDir = 'C:\Users\Usuario\Desktop\BIALYSTOK\DATASET INICIAL\SCC_PLASKONABLONKOWY-20210917T085736Z-001';

%% ==== INITIALISE STORAGE =====================================
records = []; 

%% ==== LIST SUBTYPES =======================
subtypes = dir(rootDir);
subtypes = subtypes([subtypes.isdir]);                
subtypes = subtypes(~ismember({subtypes.name}, {'.','..'}));

for s = 1:numel(subtypes)
    subtypeName = subtypes(s).name;
    subtypePath = fullfile(rootDir, subtypeName);

    % ---- LIST PATIENTS ---------------------------------------
    patients = dir(subtypePath);
    patients = patients([patients.isdir]);
    patients = patients(~ismember({patients.name}, {'.','..'}));

    for p = 1:numel(patients)
        patientName = patients(p).name;
        patientPath = fullfile(subtypePath, patientName);

        % ---- LIST SEQUENCES (subfolders inside each patient) --
        sequences = dir(patientPath);
        sequences = sequences([sequences.isdir]);
        sequences = sequences(~ismember({sequences.name}, {'.','..'}));

        for q = 1:numel(sequences)
            seqName = sequences(q).name;
            seqPath = fullfile(patientPath, seqName);

            % Find DICOM files inside this sequence folder
            dcmFiles = dir(fullfile(seqPath, '*.dcm'));
            % If no .dcm extension, try any file 
            if isempty(dcmFiles)
                allFiles = dir(seqPath);
                allFiles = allFiles(~[allFiles.isdir]);
                dcmFiles = allFiles;
            end

            if isempty(dcmFiles)
                % No images in this folder, skip
                continue;
            end

            % Read metadata from the first DICOM file in the folder
            firstFile = fullfile(seqPath, dcmFiles(1).name);
            try
                info = dicominfo(firstFile);
            catch
                % If the file is not a valid DICOM, skip this sequence
                warning('Could not read DICOM info from: %s', firstFile);
                continue;
            end

            % ---- Extract metadata with safe checks -------------
            numSlices = numel(dcmFiles);

            if isfield(info, 'Modality')
                modality = string(info.Modality);
            else
                modality = "";
            end

            if isfield(info, 'Rows')
                rowsVal = info.Rows;
            else
                rowsVal = NaN;
            end

            if isfield(info, 'Columns')
                colsVal = info.Columns;
            else
                colsVal = NaN;
            end

            pixelSpacingX = NaN;
            pixelSpacingY = NaN;
            if isfield(info, 'PixelSpacing')
                try
                    pixelSpacingX = info.PixelSpacing(1);
                    pixelSpacingY = info.PixelSpacing(2);
                catch
                end
            end

            if isfield(info, 'SliceThickness')
                sliceThickness = info.SliceThickness;
            else
                sliceThickness = NaN;
            end

            if isfield(info, 'SpacingBetweenSlices')
                spacingBetweenSlices = info.SpacingBetweenSlices;
            else
                spacingBetweenSlices = NaN;
            end

            % ---- Store record in a struct ----------------------
            rec = struct();
            rec.Subtype                   = string(subtypeName);
            rec.Patient                   = string(patientName);
            rec.Sequence                  = string(seqName);
            rec.Modality                  = modality;
            rec.Rows                      = rowsVal;
            rec.Columns                   = colsVal;
            rec.NumSlices                 = numSlices;
            rec.PixelSpacingX_mm          = pixelSpacingX;
            rec.PixelSpacingY_mm          = pixelSpacingY;
            rec.SliceThickness_mm         = sliceThickness;
            rec.SpacingBetweenSlices_mm   = spacingBetweenSlices;

            records = [records; rec]; 
        end
    end
end

%% ==== CONVERT TO TABLE AND SAVE (FULL METADATA) =============
if isempty(records)
    warning('No valid DICOM metadata were found in the specified root directory.');
else
    % Full table
    metadataadc = struct2table(records);
    disp(metadataadc);

    % Save full table
    outFileFull = fullfile(rootDir, 'metadataadc_full.csv');
    writetable(metadataadc, outFileFull);
    fprintf('Full metadata table saved to:\n%s\n', outFileFull);

    %% ==== SUMMARY BY SEQUENCE (MEDIAN + RANGE) ===============
    T = metadataadc;

    excludePETMR = true;
    valid = T.Rows > 0;
    if excludePETMR
        valid = valid & ~strcmp(T.Sequence, "PET_MR");
    end

    seqs = unique(T.Sequence);
    summary = table();

    for i = 1:numel(seqs)
        seq = seqs(i);
        idx = strcmp(T.Sequence, seq) & valid;

        if ~any(idx)
            continue;
        end

        rowsVals  = T.Rows(idx);
        colsVals  = T.Columns(idx);
        psx       = T.PixelSpacingX_mm(idx);
        psy       = T.PixelSpacingY_mm(idx);
        thick     = T.SliceThickness_mm(idx);
        slices    = T.NumSlices(idx);

        row = table;
        row.Sequence               = seq;

        % Matrix size
        row.MedianRows             = median(rowsVals);
        row.MinRows                = min(rowsVals);
        row.MaxRows                = max(rowsVals);

        row.MedianColumns          = median(colsVals);
        row.MinColumns             = min(colsVals);
        row.MaxColumns             = max(colsVals);

        % Pixel spacing
        row.MedianPixelX_mm        = median(psx, 'omitnan');
        row.MinPixelX_mm           = min(psx, [], 'omitnan');
        row.MaxPixelX_mm           = max(psx, [], 'omitnan');

        row.MedianPixelY_mm        = median(psy, 'omitnan');
        row.MinPixelY_mm           = min(psy, [], 'omitnan');
        row.MaxPixelY_mm           = max(psy, [], 'omitnan');

        % Slice thickness
        row.MedianThickness_mm     = median(thick, 'omitnan');
        row.MinThickness_mm        = min(thick, [], 'omitnan');
        row.MaxThickness_mm        = max(thick, [], 'omitnan');

        % Number of slices
        row.MinSlices              = min(slices);
        row.MaxSlices              = max(slices);

        summary = [summary; row];
    end

    %% ==== CREATE STRING COLUMNS FOR THE TFG ==================
    % Matrix size as "medianRows × medianColumns"
    summary.MatrixSize_median = strcat( ...
        string(summary.MedianRows), " × ", string(summary.MedianColumns));

    % Matrix size range as "minRows–maxRows × minCols–maxCols"
    summary.MatrixSize_range = strcat( ...
        string(summary.MinRows),  "–", string(summary.MaxRows),  " × ", ...
        string(summary.MinColumns),"–", string(summary.MaxColumns));

    % Pixel spacing as "median [min–max]" (X spacing)
    summary.PixelSpacing_median_range = strcat( ...
        string(round(summary.MedianPixelX_mm, 3)), " [", ...
        string(round(summary.MinPixelX_mm, 3)),   "–", ...
        string(round(summary.MaxPixelX_mm, 3)),   "]");

    % Slice thickness as "median [min–max]"
    summary.Thickness_median_range = strcat( ...
        string(round(summary.MedianThickness_mm, 3)), " [", ...
        string(round(summary.MinThickness_mm, 3)),    "–", ...
        string(round(summary.MaxThickness_mm, 3)),    "]");

    % Nº slices as "min–max"
    summary.NumSlices_range = strcat( ...
        string(summary.MinSlices), "–", string(summary.MaxSlices));

    %% ==== SHOW AND SAVE SUMMARY ==============================
    disp(summary);

    outFileSummary = fullfile(rootDir, 'metadataadc_summary_by_sequence_median_range.csv');
    writetable(summary, outFileSummary);
    fprintf('Summary table (median + range) saved to:\n%s\n', outFileSummary);
end

%% === COMPACT SUMMARY TABLE FOR TFG ===========================
T = metadataadc;

% 1) Keep only rows with spatial info (Rows > 0)
%    PET_MR has Rows > 0, so it will be included 
valid = T.Rows > 0;
T = T(valid, :);

% 2) Unique sequences 
seqs = unique(T.Sequence);

compact = table();

for i = 1:numel(seqs)
    seq = seqs(i);
    idx = strcmp(T.Sequence, seq);

    if ~any(idx)
        continue;
    end

    rowsVals  = T.Rows(idx);
    colsVals  = T.Columns(idx);
    psx       = T.PixelSpacingX_mm(idx);
    thick     = T.SliceThickness_mm(idx);
    slices    = T.NumSlices(idx);

    % Median & range
    medRows   = median(rowsVals);
    minRows   = min(rowsVals);
    maxRows   = max(rowsVals);

    medCols   = median(colsVals);
    minCols   = min(colsVals);
    maxCols   = max(colsVals);

    medPsx    = median(psx, 'omitnan');
    minPsx    = min(psx, [], 'omitnan');
    maxPsx    = max(psx, [], 'omitnan');

    medThick  = median(thick, 'omitnan');
    minThick  = min(thick, [], 'omitnan');
    maxThick  = max(thick, [], 'omitnan');

    minSlices = min(slices);
    maxSlices = max(slices);

 
    seqRow = table;
    seqRow.Sequence = seq;

    % Matrix size: "medianRows×medianCols [minRows–maxRows × minCols–maxCols]"
    seqRow.MatrixSize = strcat( ...
        string(medRows), "×", string(medCols), ...
        " [", string(minRows), "–", string(maxRows), ...
        " × ", string(minCols), "–", string(maxCols), "]" );

    % Pixel spacing: if all NaN, mark as "Not available"
    if all(isnan(psx))
        seqRow.PixelSpacing_mm = "Not available";
    else
        seqRow.PixelSpacing_mm = strcat( ...
            string(round(medPsx, 3)), " [", ...
            string(round(minPsx, 3)), "–", ...
            string(round(maxPsx, 3)), "]" );
    end

    % Slice thickness: if all NaN, mark as "Not available"
    if all(isnan(thick))
        seqRow.SliceThickness_mm = "Not available";
    else
        seqRow.SliceThickness_mm = strcat( ...
            string(round(medThick, 3)), " [", ...
            string(round(minThick, 3)), "–", ...
            string(round(maxThick, 3)), "]" );
    end

    % Nº of slices: "min–max"
    seqRow.NumSlices = strcat( ...
        string(minSlices), "–", string(maxSlices) );

    compact = [compact; seqRow]; 
end

% 3) Show compact summary in MATLAB
disp(compact);

% 4) Save to CSV 
outFileCompact = fullfile(rootDir, 'metadataadc_compact_summary_for_TFG.csv');
writetable(compact, outFileCompact);
fprintf('Compact summary table saved to:\n%s\n', outFileCompact);
