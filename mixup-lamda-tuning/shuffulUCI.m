function shuffulUCI(dataset)
datapath = strcat('DATASET/',dataset,'.dat');
data = readtable(datapath);
rng(9);
data_shuffled = data(randperm(size(data,1)), :);
writetable(data_shuffled,datapath);