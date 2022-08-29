function [train_examples,train_labels,test_examples,test_labels] = getdataUCI(data,k,i)
%ʹ��cross validationd��ʽ����ѵ�����Ͳ��Լ���
%����dataset��ʾ�ļ�����k��ʾ�����ݼ���Ϊ���ݣ�i��ʾѡ�����е�i����Ϊ���Լ�������Ϊѵ����
%rng(9);
%data_shuffled = data(randperm(size(data,1)), :);
%writetable(data_shuffled,datapath);
[n,m] = size(data);
%ÿһ����ntest��
nTest = floor(n/k);
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
test_labels = categorical(data_test{:,'clase'});
test_examples = data_test;
test_examples(:,'clase') = [];
test_examples(:,1) = [];
% separate the examples and the labels for the training dataset:
train_labels = categorical(data_train{:,'clase'});
train_examples = data_train;
train_examples(:,'clase') = [];
train_examples(:,1) = [];

end
