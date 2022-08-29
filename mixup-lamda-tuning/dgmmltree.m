classdef dgmmltree
    methods(Static)
        function m = fit(train_examples, train_labels,j, minleafsize)
            
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
            m.min_parent_size = minleafsize;
            m.unique_classes = unique(r.labels);
            m.feature_names = train_examples.Properties.VariableNames;
			m.nodes = 1;
            m.index = 1;
            m.N = size(train_examples,1);
            m.tree = dgmmltree.trySplit(m, r,j);

        end
        
        function node = trySplit(m, node,rndi)
                if size(node.examples, 1) <= m.min_parent_size
                    return
                end
                if length(unique(node.labels)) == 1
                    return
                end         
                %����ڵ���ֻ��һ�����������ݣ�������ݿ������쳣���ݣ�Ҳֹͣ����
                examples = node.examples;
                examples = table2array(examples);
                label = unique(node.labels);
                k = size(label,1);
                X1 = examples(find(node.labels==label(1)),:);
                X2 = examples(find(node.labels==label(2)),:);
                x1=size(X1,1);
                x2=size(X2,1);
                %����mtry��������������ѡ��mtry��feature candicate
                %������������ӣ������ýڵ�����ʾ
                rng(m.nodes*rndi);
                feature_num = size(node.examples,2);
                if feature_num >10
                    mtry = round(sqrt(feature_num));
                else
                    mtry = feature_num;
                end
                new_index = randperm(feature_num,mtry);
                X1 = X1(:,new_index);
                X2 = X2(:,new_index);
                m1 = mean(X1,1);
                m2 = mean(X2,1);
                m1_mat = repmat(m1,x1,1);
                m2_mat = repmat(m2,x2,1);
                %ԭ���Ĵ��������m2_mat = repmat(m1,x2,1);
                x1_m1 = (X1-m1_mat).^2;
                x2_m2 = (X2-m2_mat).^2;
                m1_m2 = (m1-m2).^2;
                sum1 = sum(x1_m1,1);
                sum2 = sum(x2_m2,1);
                denominator = sum1 + sum2;
                %�����ĸΪ0������ӷ�ĸ����һ
                k = find(denominator == 0);
                denominator(1,k) = 1;
                m1_m2(1,k) = m1_m2(1,k) + 1;
                w1 = m1_m2 ./ denominator;
                w1 = w1.^(1/2);
                [value,winning_feature1] = max(w1);
                winning_feature = new_index(winning_feature1);
                [ps,n] = sortrows(node.examples,winning_feature);
                ls = node.labels(n);
                %�����ŵķ���ֵ
                %[va,winning_index] = min(abs(mean_feature_value-featurevalue))
                num = size(node.examples,1);
                winning_index = floor(size(node.examples,1)/2);
                node.splitFeature = winning_feature;
               % node.splitFeatureName = m.feature_names{winning_feature};
                node.splitValue = (ps{winning_index,winning_feature} + ps{winning_index+1,winning_feature}) / 2;

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
                
                node.children{1} = dgmmltree.trySplit(m, node.children{1},rndi);
                node.children{2} = dgmmltree.trySplit(m, node.children{2},rndi);

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
               dgmmltree.describeNode(node.children{1});
               dgmmltree.describeNode(node.children{2});        
            end
		end
        
        function predictions = predict(m, test_examples)

            predictions = categorical;
            
            for i=1:size(test_examples,1)
                
				%fprintf('classifying example %i/%i\n', i, size(test_examples,1));
                this_test_example = test_examples{i,:};
                this_prediction = dgmmltree.predict_one(m, this_test_example);
                predictions(end+1) = this_prediction;
			end
        end

        function prediction = predict_one(m, this_test_example)
            
			node = dgmmltree.descend_tree(m.tree, this_test_example);
            prediction = node.prediction;
        
		end
        
        function node = descend_tree(node, this_test_example)
            
			if isempty(node.children)
                return;
            else
                if this_test_example(node.splitFeature) < node.splitValue
                    node = dgmmltree.descend_tree(node.children{1}, this_test_example);
                else
                    node = dgmmltree.descend_tree(node.children{2}, this_test_example);
                end
            end
        end
   end
end