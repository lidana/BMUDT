classdef carttree
    methods(Static)
        
        function m = fit(train_examples, train_labels,j, minleafsize)
            
			emptyNode.number = [];
            emptyNode.examples = [];
            emptyNode.labels = [];
            emptyNode.prediction = [];
            emptyNode.impurityMeasure = [];
            emptyNode.children = {};
            emptyNode.splitFeature = [];
            emptyNode.splitFeatureName = [];
            emptyNode.splitValue = [];

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

            m.tree = carttree.trySplit(m, r,j);

        end
        
        function node = trySplit(m, node,j)

            if size(node.examples, 1) <= m.min_parent_size
				return
            end
            if length(unique(node.labels)) == 1
                return
            end 

            node.impurityMeasure = carttree.weightedImpurity(m, node.labels);
            rng(m.nodes*j);
            feature_num = size(node.examples,2);
            mtry = round(sqrt(feature_num));
            new_index = randperm(feature_num,mtry);
            examples = node.examples(:,new_index);
            for i=1:1:mtry
			%	fprintf('evaluating possible splits on feature %d/%d\n', new_index(i), size(node.examples,2));
				[ps,n] = sortrows(examples,i);
                ls = node.labels(n);
                biggest_reduction(i) = -Inf;
                biggest_reduction_index(i) = -1;
                biggest_reduction_value(i) = NaN;
                for j=1:(size(ps,1)-1)
                    if ps{j,i} == ps{j+1,i}
                        continue;
                    end
                    
                    this_reduction = node.impurityMeasure - (carttree.weightedImpurity(m, ls(1:j)) + carttree.weightedImpurity(m, ls((j+1):end)));
                    
                    if this_reduction > biggest_reduction(i)
                        biggest_reduction(i) = this_reduction;
                        biggest_reduction_index(i) = j;
                    end
                end
				
            end

            [winning_reduction,winning_feature] = max(biggest_reduction);
            winning_index = biggest_reduction_index(winning_feature);
            winning_feature = new_index(winning_feature);
            if winning_reduction <= 0
                return
            else
                [ps,n] = sortrows(node.examples,winning_feature);
                ls = node.labels(n);

                node.splitFeature = winning_feature;
                node.splitFeatureName = m.feature_names{winning_feature};
                node.splitValue = (ps{winning_index,winning_feature} + ps{winning_index+1,winning_feature}) / 2;

                node.examples = [];
                node.labels = []; 
                node.prediction = [];

                node.children{1} = m.emptyNode;
                m.nodes = m.nodes + 1; 
                node.children{1}.number = m.nodes;
                node.children{1}.examples = ps(1:winning_index,:); 
                node.children{1}.labels = ls(1:winning_index);
                node.children{1}.prediction = mode(node.children{1}.labels);
                
                node.children{2} = m.emptyNode;
                m.nodes = m.nodes + 1; 
                node.children{2}.number = m.nodes;
                node.children{2}.examples = ps((winning_index+1):end,:); 
                node.children{2}.labels = ls((winning_index+1):end);
                node.children{2}.prediction = mode(node.children{2}.labels);
                
                node.children{1} = carttree.trySplit(m, node.children{1},j);
                node.children{2} = carttree.trySplit(m, node.children{2},j);
            end

        end
        
        function e = weightedImpurity(m, labels)

            weight = length(labels) / m.N;

            summ = 0;
            obsInThisNode = length(labels);
            for i=1:length(m.unique_classes)
                
				pc = length(labels(labels==m.unique_classes(i))) / obsInThisNode;
                summ = summ + (pc*pc);
            
			end
            g = 1 - summ;
            
            e = weight * g;

        end

        function predictions = predict(m, test_examples)

            predictions = categorical;
            
            for i=1:size(test_examples,1)
                
				%fprintf('classifying example %i/%i\n', i, size(test_examples,1));
                this_test_example = test_examples{i,:};
                this_prediction = carttree.predict_one(m, this_test_example);
                predictions(end+1) = this_prediction;
            
			end
        end

        function prediction = predict_one(m, this_test_example)
            
			node = carttree.descend_tree(m.tree, this_test_example);
            prediction = node.prediction;
        
		end
        
        function node = descend_tree(node, this_test_example)
            
			if isempty(node.children)
                return;
            else
                if this_test_example(node.splitFeature) < node.splitValue
                    node = carttree.descend_tree(node.children{1}, this_test_example);
                else
                    node = carttree.descend_tree(node.children{2}, this_test_example);
                end
            end
        
		end
        
        % describe a tree:
        function describeNode(node)
            
			if isempty(node.children)
                fprintf('Node %d; %s\n', node.number, node.prediction);
            else
                fprintf('Node %d; if %s <= %f then node %d else node %d\n', node.number, node.splitFeatureName, node.splitValue, node.children{1}.number, node.children{2}.number);
                carttree.describeNode(node.children{1});
                carttree.describeNode(node.children{2});        
            end
        
		end
		
    end
end