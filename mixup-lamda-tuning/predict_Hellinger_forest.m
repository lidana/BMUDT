function [predicted_classes,predicted_scores] = predict_Hellinger_forest(model,features)
%Function: predict_Hellinger_forest
%Form: predicted_classes = predict_Hellinger_forest(model,features)
%Description: Predict labels using trained Hellinger Distance Decision
%       Forest
%Parameters:
%   model: a trained Hellinger Distance Decision Forest model
%   features: I X F numeric matrix where I is the number of instances and F
%       is the number of features. Each row represents one training instance
%       and each column represents the value of one of its corresponding features
%Output:
%   predicted_classes: I X 1 matrix where each row represents a predicted label of the corresponding feature set 
%   predicted_scores: I X 1 matrix where each row represents a
%   score of the corresponding feature set having label "1"/"true"/"positive".

[numInstances,numFeatures] = size(features);

if numInstances <= 0
    msgID = 'predict_Hellinger_forest:notEnoughData';
    msg = 'Feature array is empty or only one instance exists';
    causeException = MException(msgID,msg);
    throw(causeException);
end

if numFeatures == 0
    msgID = 'predict_Hellinger_forest:noData';
    msg = 'No feature data';
    causeException = MException(msgID,msg);
    throw(causeException);
end

numTrees = size(model,1);
predictions = zeros(size(features,1),numTrees);
scores = zeros(size(features,1),numTrees);
for i = 1:1:numTrees
    [predictions(:,i),scores(:,i)] = predict_Hellinger_tree(model{i,1},features(:,model{i,2}));
end
predicted_classes = mode(predictions,2);
predicted_scores = mean(scores,2);

end