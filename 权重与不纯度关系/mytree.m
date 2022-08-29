classdef mytree
    methods(Static)
        
        function m = fit(train_examples, train_labels,j)
            
			emptyNode.number = [];
            emptyNode.examples = [];
            emptyNode.labels = [];
            emptyNode.prediction = [];
            emptyNode.children = {};
            emptyNode.A = [];
            m.emptyNode = emptyNode;
            
            r = emptyNode;
            r.number = 1;
            r.index = 1;
            r.labels = train_labels;
            r.examples = train_examples;
            r.prediction = mode(r.labels);
            m.min_parent_size = 10;
            m.unique_classes = unique(r.labels);
            m.feature_names = train_examples.Properties.VariableNames;
			m.nodes = 1;
            m.index = 1;
            m.N = size(train_examples,1);
            m.tree = mytree.trySplit(m, r,j);

        end
        
        function node = trySplit(m, node,j)
            if size(node.examples, 1) < m.min_parent_size
				return
            end
            if length(unique(node.labels)) == 1
                return
            end         
            %如果节点中只有一个不纯的数据，这个数据可能是异常数据，也停止分裂
            la = unique(node.labels);
            for i = 1:1:length(la)
            id = find(node.labels == la(i));
            ni = length(id);
            if ni==1&&ni/size(node.examples,1)<=0.1
             %   return
            end
            end
            examples = node.examples;
            examples = table2array(examples);
            
            label = unique(node.labels);
            k = size(label,1);
            X1 = examples(find(node.labels==label(1)),:);
            X2 = examples(find(node.labels==label(2)),:);
            x1=size(X1,1);
            x2=size(X2,1);
            %计算mtry，从所有特征中选出mtry个feature candicate
            %重置随机数种子，可以用节点数表示
            rng(m.nodes*j);
            feature_num = size(node.examples,2);
            %mtry = round(sqrt(feature_num));
            mtry = feature_num;
            new_index = randperm(feature_num,mtry);
            X1 = X1(:,new_index);
            X2 = X2(:,new_index);
            m1 = mean(X1,1);
            m2 = mean(X2,1);
            for j = 1:1:mtry
                numerator =power((m1(1,j) - m2(1,j)),2);
                sum1 = 0;
                sum2 = 0;
                for i =1:1:x1
                    sum1 = sum1 + power(X1(i,j)-m1(1,j),2);
                end
                 for i =1:1:x2
                    sum2 = sum2 + power(X2(i,j)-m2(1,j),2);
                end
                denominator = sum1 + sum2;
                if denominator == 0
                     w1(j) = sqrt((numerator+1)/(denominator+1));
                else
                     w1(j) = sqrt(numerator/denominator);
                end
            end
                [value,winning_feature1] = max(w1);
                winning_feature = new_index(winning_feature1);
                [ps,n] = sortrows(node.examples,winning_feature);
                ls = node.labels(n);
                %找最优的分裂值
                %[va,winning_index] = min(abs(mean_feature_value-featurevalue))
                num = size(node.examples,1);
                winning_index = floor(size(node.examples,1)/2);

                 %找最佳分裂值，找到两类数据在最优特征上的最大值和最小值
                 %（防止异常值的出现，如果节点数量足够，就用排在前五个的值的均值表示最大值）
                 la = unique(node.labels);
                 cate1 = table2array(ps(ls==la(1),:));
                 cate2 = table2array(ps(ls==la(2),:));
                 if size(cate1,1)>10
                     maxvalue5 = cate1(end-5:1:end,winning_feature);
                     maxvalue_cate1 = mean(maxvalue5);
                     minvalue5 = cate1(1:1:5,winning_feature);
                     minvalue_cate1 = mean(minvalue5);
                 else
                     maxvalue_cate1 = cate1(end,winning_feature);
                     minvalue_cate1 = cate1(1,winning_feature);
                 end
                 if size(cate2,1)>10
                     maxvalue5 = cate2(end-5:1:end,winning_feature);
                     maxvalue_cate2 = mean(maxvalue5);
                     minvalue5 = cate2(1:1:5,winning_feature);
                     minvalue_cate2 = mean(minvalue5);
                 else
                     maxvalue_cate2 = cate2(end,winning_feature);
                     minvalue_cate2 = cate2(1,winning_feature);
                 end
                %如果一类的最大值大于二类的最大值，表明一类在右边
                %如果只考虑最大值，会出现最大值的确在前面，但是其实类在后面的情况，可以考虑两类的均值的大小
                flag1 = mean(cate1(:,winning_feature),1);
                flag2 = mean(cate2(:,winning_feature),1);
               % if maxvalue_cate1>maxvalue_cate2
                if flag1>flag2
                    splitvalue = (minvalue_cate1+maxvalue_cate2)/2;
                else
                    splitvalue = (minvalue_cate2+maxvalue_cate1)/2;
                end   
                [~,winning_index] = min(abs(table2array(ps(:,winning_feature))-splitvalue));
                if winning_index == num
                    return
                end
                node.splitFeature = winning_feature;
                node.splitFeatureName = m.feature_names{winning_feature};
                node.splitValue = (ps{winning_index,winning_feature} + ps{winning_index+1,winning_feature}) / 2;

               % node.examples = [];
                %node.labels = []; 
                node.prediction = [];

                node.children{1} = m.emptyNode;
                m.nodes = m.nodes + 1; 
                m.index = m.index*2;
                node.children{1}.number = m.nodes;
                node.children{1}.index = m.index;
                node.children{1}.examples = ps(1:winning_index,:); 
                node.children{1}.labels = ls(1:winning_index);
                node.children{1}.prediction = mode(node.children{1}.labels);
                
                node.children{2} = m.emptyNode;
                m.nodes = m.nodes + 1; 
                m.index = m.index*2+1;
                node.children{2}.index = m.index;
                node.children{2}.number = m.nodes;
                node.children{2}.examples = ps((winning_index+1):end,:); 
                node.children{2}.labels = ls((winning_index+1):end);
                node.children{2}.prediction = mode(node.children{2}.labels);
                
                node.children{1} = mytree.trySplit(m, node.children{1},j);
                node.children{2} = mytree.trySplit(m, node.children{2},j);

        end
        % describe a tree:
         function describeNode(node)
			if isempty(node.children)
                fprintf('Node %d; %s\n', node.number, node.prediction);
                X1 = table2array(node.examples);
                X = tsne(X1);
                figure;
                gscatter(X(:,1),X(:,2),node.labels);
            else
               fprintf('Node %d; if %s <= %f then node %d else node %d\n', node.number, node.splitFeatureName, node.splitValue, node.children{1}.number, node.children{2}.number);
                num = size(node.examples,1); 
               figure;
               Y = zeros(1,num);
               X = node.examples(:,node.splitFeature);
               num2 = size(X,1);
               fprintf('Node example number %d,figure example number %d\n',num,num2);         
                hAxes = axes('NextPlot','add',...           %# Add subsequent plots to the axes,
               'DataAspectRatio',[1 1 1],...  %#   match the scaling of each axis,
               'XLim',[0 1],...               %#   set the x axis limit,
               'YLim',[0 eps],...             %#   set the y axis limit (tiny!),
               'Color','none');  
               hold on;
               X = table2array(X);
               X = X';
               gscatter(X,Y,node.labels);
               gscatter(node.splitValue,0,'SplitValue','k');    
               legend off;
               legend show;   
               set(legend,'Location','best');
               xlabel('SplitFeature');
               hold off;
               mytree.describeNode(node.children{1});
               mytree.describeNode(node.children{2});        
            end
		end
        
        function predictions = predict(m, test_examples)

            predictions = categorical;
            
            for i=1:size(test_examples,1)
                
				%fprintf('classifying example %i/%i\n', i, size(test_examples,1));
                this_test_example = test_examples{i,:};
                this_prediction = mytree.predict_one(m, this_test_example);
                predictions(end+1) = this_prediction;
			end
        end

        function prediction = predict_one(m, this_test_example)
            
			node = mytree.descend_tree(m.tree, this_test_example);
            prediction = node.prediction;
        
		end
        
        function node = descend_tree(node, this_test_example)
            
			if isempty(node.children)
                return;
            else
                if this_test_example(node.splitFeature) < node.splitValue
                    node = mytree.descend_tree(node.children{1}, this_test_example);
                else
                    node = mytree.descend_tree(node.children{2}, this_test_example);
                end
            end
        end
   end
end