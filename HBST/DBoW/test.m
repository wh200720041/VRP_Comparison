clc;
close all;

sceneImage = rgb2gray(imread('C:\Users\IceHan\Desktop\0000003079.png'));
boxImage = rgb2gray(imread('C:\Users\IceHan\Desktop\0000002420.png'));
boxPoints = detectFASTFeatures(boxImage);
scenePoints = detectFASTFeatures(sceneImage);


[boxFeatures, boxPoints] = extractFeatures(boxImage, boxPoints,'Method','BRISK');
[sceneFeatures, scenePoints] = extractFeatures(sceneImage, scenePoints,'Method','BRISK');


A=zeros(300,1000);
B=ones(300,1000);
tic
for i=1:1000
    matchFeatures(boxFeatures, sceneFeatures,'MaxRatio',0.9);
end
toc


%{
figure;
a = insertMarker(sceneImage,scenePoints,'circle');
imshow(a)
%}

%{
boxPairs = matchFeatures(boxFeatures, sceneFeatures,'MaxRatio',0.9);

matchedBoxPoints = boxPoints(boxPairs(:, 1), :);
matchedScenePoints = scenePoints(boxPairs(:, 2), :);



figure;
showMatchedFeatures(boxImage, sceneImage, matchedBoxPoints, ...
    matchedScenePoints, 'montage');
title('Putatively Matched Points (Including Outliers)');
%}
%{
[tform, inlierBoxPoints, inlierScenePoints] = ...
    estimateGeometricTransform(matchedBoxPoints, matchedScenePoints, 'affine');
figure;
showMatchedFeatures(boxImage, sceneImage, inlierBoxPoints, ...
    inlierScenePoints, 'montage');
title('Matched Points (Inliers Only)');
%}
