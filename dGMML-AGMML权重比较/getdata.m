function [train_examples,train_labels,test_examples,test_labels] = getdata(dataset,k,i)
%使用cross validationd方式返回训练集和测试集，
%参数dataset表示文件名，k表示将数据集分为几份，i表示选择其中第i个作为测试集，其他为训练集
datapath = strcat('DATAOPENML/',dataset,'.csv');
data = readtable(datapath);
%rng(9);
%data_shuffled = data(randperm(size(data,1)), :);
%writetable(data_shuffled,datapath);
[n,m] = size(data);
%每一份有ntest个
nTest = round(n/k);
if i == k 
    data_test = data(((i-1)*nTest+1):end,:);
    data_train = data(1:(i-1)*nTest,:);
else
    data_test = data(((i-1)*nTest+1):1:i*nTest, :);
    data_first = data(1:1:(i-1)*nTest,:);
    data_end = data((i*nTest+1):1:end,:);
    data_train = [data_first;data_end];
end
% separate the examples and the labels for the testing dataset:
test_labels = categorical(data_test{:,end});
test_examples = data_test;
test_examples(:,end) = [];
% separate the examples and the labels for the training dataset:
train_labels = categorical(data_train{:,end});
train_examples = data_train;
train_examples(:,end) = [];

end