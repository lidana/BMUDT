classdef mytree
    methods(Static)
        
        function m = fit(train_examples, train_labels, minleafsize)
            
			emptyNode.number = [];
            emptyNode.examples = [];
            emptyNode.labels = [];
            emptyNode.prediction = [];
            emptyNode.children = {};
            %�ֱ��¼�����������ӽڵ�ľ������ģ����ӽڵ�ľ�������
            emptyNode.A = [];
            emptyNode.lc = [];
            emptyNode.rc = [];

            m.emptyNode = emptyNode;
            
            r = emptyNode;
            r.number = 1;
            r.labels = train_labels;
            r.examples = train_examples;
            r.prediction = mode(r.labels);
            m.min_parent_size = minleafsize;
            m.unique_classes = unique(r.labels);
            m.feature_names = train_examples.Properties.VariableNames;
			m.nodes = 1;
            m.N = size(train_examples,1);
            
            t = 0.5;
            AA = MetricLearning(train_examples,train_labels, t);
            m.A = AA;
            m.tree = mytree.trySplit(m, r);

        end
        
        
        function node = trySplit(m, node)

            if size(node.examples, 1) < m.min_parent_size
				return
            end
            if size(unique(node.labels)) == 1
                return
            end            
            %ʹ��GMML����ýڵ����ݵĶ�������A��������������ݵ����ģ�������1���������ĵ����Ͼ���С�ͷֵ���ڵ㣬�����ҽڵ� 
            %  AA = LRGMML(node.examples,node.labels)
            %������������������
            label = unique(node.labels);
            k = size(label,1);
            n = size(node.examples,1);
            
            mean_x1 = mean(node.examples{node.labels==label(1),:});
            mean_x2 = mean(node.examples{node.labels==label(2),:});

            index1 = 1;
            index2 = 1;
            for i = 1:1:n
                dx1 = (node.examples{i,:}-mean_x1)*m.A*(node.examples{i,:}-mean_x1)';
                dx2 = (node.examples{i,:}-mean_x2)*m.A*(node.examples{i,:}-mean_x2)';
                if dx1<=dx2
                    node1(index1,:) = node.examples(i,:);
                    node1_label(index1,:) = node.labels(i);
                    index1 = index1+1;
                else  %dx1>dx2
                    node2(index2,:) = node.examples(i,:);
                    node2_label(index2,:) = node.labels(i);
                    index2 = index2+1;    
                end
            end
            %�������֮�󶼻��ֵ�ͬһ�ߣ�index1=1����index2=1���������û��֣�ֹͣ����
            if index1==1 | index2==1
                return
            end
            
                node11 = table2array(node1);
                node22 = table2array(node2);
                llc = mean(node11,1);
                rrc = mean(node22,1);   
                node.examples = [];
                node.labels = []; 
                node.prediction = [];
                node.lc = llc;
                node.rc = rrc;

                node.children{1} = m.emptyNode;
                m.nodes = m.nodes + 1; 
                node.children{1}.number = m.nodes;
                node.children{1}.examples = node1; 
                node.children{1}.labels = node1_label;
                node.children{1}.prediction = mode(node.children{1}.labels);
                
                node.children{2} = m.emptyNode;
                m.nodes = m.nodes + 1; 
                node.children{2}.number = m.nodes;
                node.children{2}.examples = node2; 
                node.children{2}.labels = node2_label;
                node.children{2}.prediction = mode(node.children{2}.labels);
                
                node.children{1} = mytree.trySplit(m, node.children{1});
                node.children{2} = mytree.trySplit(m, node.children{2});

        end
        % describe a tree:
        function describeNode(node)
			if isempty(node.children)
                fprintf('Node %d; %s\n', node.number, node.prediction);
               % if size(node.examples,1)>=1
               % X1 = table2array(node.examples);
               % X = tsne(X1);%,'Distance','mahalanobis');
               % figure;
               % gscatter(X(:,1),X(:,2),node.labels);
               % end
            else
                fprintf('Node %d: cluster center ',node.number);
                disp(node.lc)
                fprintf('-->node %d; cluster center ' ,node.children{1}.number);
                disp(node.rc)
                fprintf('-->node %d', node.children{2}.number);
              %  if size(node.examples,1)>=1
              %  node.examples
               % X1 = table2array(node.examples);
               % X = tsne(X1);%,'Distance','mahalanobis');
               % figure;
               % gscatter(X(:,1),X(:,2),node.labels);
               %end
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
            
			node = mytree.descend_tree(m.tree, m, this_test_example);
            prediction = node.prediction;
        
		end
        
        function node = descend_tree(node, m, this_test_example)
            
			if isempty(node.children)
                return;
            else
                dis_left = (this_test_example-node.lc)*m.A*(this_test_example-node.lc)';
                dis_right = (this_test_example-node.rc)*m.A*(this_test_example-node.rc)';
                if dis_left < dis_right
                    node = mytree.descend_tree(node.children{1}, m, this_test_example);
                else
                    node = mytree.descend_tree(node.children{2}, m, this_test_example);
                end
            end
        end
   end
end