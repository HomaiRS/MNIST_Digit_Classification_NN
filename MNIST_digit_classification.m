% clc; clear all; close all;
function [TrainImg, TrainLabel, TestImg, TestLabel] = MNIST_digit_classification(Train_num_image, Test_num_image)
% Train_num_image = 60000; Test_num_image = 30;

TrainImg   = ReadBinaryImages('train-images-idx3-ubyte',Train_num_image);
disp('-----------HW2_MNIST code: Successfully read TrainImg------------')
TrainLabel = ReadBinaryLabels('train-labels-idx1-ubyte',Train_num_image);
disp('-----------HW2_MNIST code: Successfully read TrainLabel------------')

TestImg    = ReadBinaryImages('t10k-images-idx3-ubyte',Test_num_image);
disp('-----------HW2_MNIST code: Successfully read TestImg------------')
TestLabel  = ReadBinaryLabels('t10k-labels-idx1-ubyte',Test_num_image);
disp('-----------HW2_MNIST code: Successfully read TestLabel------------')

%  myImShow(TrainImg, TrainLabel)
%  myImShow(TestImg, TestLabel)
end

function labels = ReadBinaryLabels(fileName,totalImages)
offset=[0];
    % Read digit labels
    fid = fopen(fileName, 'r', 'b'); header = fread(fid, 1, 'int32');
    if header ~= 2049
        error('Invalid label file header');
    end
    count = fread(fid, 1, 'int32');
    if count < totalImages+offset, error('Trying to read too many digits'); end
    if offset > 0, fseek(fid, offset, 'cof'); end
    
    labels = fread(fid, totalImages, 'uint8'); fclose(fid);
end

%--- My binary file reader function :: Make sure to extract the files (not .gz)
function imageCellArray = ReadBinaryImages(fileName,totalImages)
    %//Open file
    fid = fopen(fileName, 'r');
    %%--- magicNumber
    A = fread(fid, 1, 'uint32');
    magicNumber = swapbytes(uint32(A));
    %%--- totalImages
    A = fread(fid, 1, 'uint32');
%     totalImages = swapbytes(uint32(A));
    %%***************Manually set (totalImages = 10) for the time being:*************
%     totalImages = 10;
    %%--- numRows
    A = fread(fid, 1, 'uint32');
    numRows = swapbytes(uint32(A));
    %%--- numCols
    A = fread(fid, 1, 'uint32');
    numCols = swapbytes(uint32(A));

    %//For each image, store into an individual cell
    imageCellArray = cell(1, totalImages);
    for k = 1 : totalImages
        %//Read in numRows*numCols pixels at a time
        A = fread(fid, numRows*numCols, 'uint8');
        %//Reshape so that it becomes a matrix
        %//We are actually reading this in column major format
        %//so we need to transpose this at the end
        imageCellArray{k} = reshape(uint8(A), numCols, numRows)';
    end
    %//Close the file
    fclose(fid);
end

function myImShow(CellArray, labels)
figure()
    for i=1:length(CellArray)
%         subplot(5,4,i)
        subplot_tight(10,10,i,[0.04,0.04])
        imshow(CellArray{i});
        title(['Labeled: ' num2str(labels(i))])
    end
end

