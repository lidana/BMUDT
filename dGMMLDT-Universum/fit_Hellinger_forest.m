function model = fit_Hellinger_forest(features, labels, numTrees, numBins, minFeatureRatio, cutoff, printCount, memSplit, memThresh)
%Function: fit_Hellinger_forest
%Form: model = fit_Hellinger_forest(features, labels, numTrees, numBins, minFeatureRatio, cutoff, printCount, memSplit, memThresh)
%Description: Train a forest of Hellinger Distance Decision Tree based on
%       random subsets of the features
%Parameters:
%   features: I X F numeric matrix where I is the number of instances and F
%       is the number of features. Each row represents one training instance
%       and each column represents the value of one of its corresponding features
%   labels: I x 1 numeric matrix where I is the number of instances. Each
%       row is the label of a specific training instance and corresponds to
%       the same row in features
%   numTrees: Number of trees to grow
%   numBins (optional): Number of bins for discretizing numeric features. 
%        Default: 100
%   minFeatureRatio (optional): Minimum percent of features to use for each split as a
%       decimal value. Default: 0.8
%   cutoff (optional): Number representing maximum number of instances in a
%       leaf node. Default: 10 if more than ten instances, 1 otherwise
%   printCount (optional): Boolean representing whether the number of current tree being grown is printed.
%       Default: false
%   memSplit (optional): If features matrix is large, compute discretization splits
%       iteratively in batches of size memSplit instead all at once. Default: 1
%   memThresh (optional): If features matrix is large, compute discretization splits
%       iteratively in batches of size memSplit only if number of instances
%       in branch is greater than memThresh. Default: 1
%Output:
%   model: a trained Hellinger Distance Decision Forest model

[numInstances,numFeatures] = size(features);

if numInstances <= 1
    msgID = 'fit_Hellinger_forest:notEnoughData';
    msg = 'Feature array is empty or only instance exists';
    causeException = MException(msgID,msg);
    throw(causeException);
end

if numFeatures == 0
    msgID = 'fit_Hellinger_forest:noData';
    msg = 'No feature data';
    causeException = MException(msgID,msg);
    throw(causeException);
end

if size(labels,1) ~= numInstances
    msgID = 'fit_Hellinger_forest:mismatchInstanceSize';
    msg = 'Number of instances in feature matrix and label matrix do not match';
    causeException = MException(msgID,msg);
    throw(causeException);
end

labelIDs = unique(labels);
if length(labelIDs) ~= 2 || ismember(0,ismember(labelIDs,[0 1]))
    msgID = 'fit_Hellinger_forest:improperLabels';
    msg = 'Labels must be either 0 or 1; Label array may only contain a single label value';
    causeException = MException(msgID,msg);
    throw(causeException);
end

if nargin < 4 || isempty(numBins)
    numBins = 100;
end

if nargin < 5 || isempty(minFeatureRatio)
    minFeatureRatio = 0.8;
end

if nargin < 6 || isempty(cutoff)
    cutoff = 10;
    if numInstances <= 10
        cutoff = 1;
    end
end

if nargin < 7 || isempty(printCount)
    printCount = false;
end

if nargin < 8 || isempty(memSplit)
    memSplit = 1;
end

if nargin < 9 || isempty(memThresh)
    memThresh = 1;
end

if numTrees < 1
    msgID = 'fit_Hellinger_forest:improperTreeCount';
    msg = 'numTrees must be 1 or larger';
    causeException = MException(msgID,msg);
    throw(causeException);
end

if minFeatureRatio > 1 || minFeatureRatio <= 0
    msgID = 'fit_Hellinger_forest:improperMinFeatureRatioRange';
    msg = 'minFeatureRatio must be between (0 and 1]';
    causeException = MException(msgID,msg);
    throw(causeException);
end

numFeatures = size(features,2);
minFeatures = ceil(minFeatureRatio .* numFeatures);

model = cell(numTrees,2);

for i = 1:1:numTrees
    if printCount
        disp(['Growing Tree Number:' num2str(i)]);
    end
    reducedFeaturesIndices = randperm(numFeatures);
    numFeaturesReduced = randi([minFeatures, numFeatures]);
    reducedFeaturesIndices = reducedFeaturesIndices(1:numFeaturesReduced);
    model{i,2} = reducedFeaturesIndices;
    reducedFeatures = features(:,reducedFeaturesIndices);
    treeModel = HellingerTreeNode;
    model{i,1} = HDDT(reducedFeatures,labels,treeModel,numBins,cutoff,memThresh,memSplit);
end

end
