classdef GenericTest
    
    properties( Access = protected )
        
        dataset;
        trainingDataset;
        hypothesesCount;
        conditionalProbabilities;
        p;
        hypothesesStrings;
        costs;
        assignedClasses;
        accuracy;
        falseAlarmProb;
        detectionProb;
        missDetectionProb;
        errorProb;
        
    end % class data members
    
    methods( Access = protected )
        
        % Constructor 
        function obj = GenericTest( completeDataset, HCount )
            
            obj.dataset = completeDataset;
            obj.trainingDataset = obj.dataset.getTrainingDatabase();
            obj.hypothesesCount = HCount;
            
            % The size depends on the number of hypotheses
            obj.conditionalProbabilities = zeros(...
                obj.dataset.getFeaturesCount, obj.hypothesesCount );
            
            obj.p = zeros( 1, obj.hypothesesCount );
            obj.hypothesesStrings = ['Park'; 'Code'; 'Sign'; 'Traf' ];
            obj.accuracy = 0;
            obj.falseAlarmProb = 0;
            obj.detectionProb = 0;
            obj.missDetectionProb = 0;
            obj.errorProb = 0;           
            
        end % constructor
        
        % Use Maximum liklihood to estimate the probability
        % of hypotheses
        function estimatedProbability = ...
                estimateProbabilityUsingML( obj, datasetOfH )
            
            totalNumberOfEntries = obj.trainingDataset.getEntriesCount();
            numberOfEntries = datasetOfH.getEntriesCount();
            estimatedProbability = numberOfEntries / totalNumberOfEntries;
            
        end % function estimateProbability
        
        % Use the given maximum likelihood estimator to
        % estimate the conditional probability for feature i
        function estimatedProbability = ...
                estimateConditionalProbUsingBernoulliModel(...
                obj, i, hypothesisDataset )
            
            Nij = hypothesisDataset.getEntriesCountForTheFeature( i );
            Nj = hypothesisDataset.getEntriesCount();
            estimatedProbability = ( Nij + 1 ) / ( Nj + 1 );            
            
        end % function estimateConditionalProbUsingBernoulliModel
        
        % Find the probability of the hypothesis assoicated with
        % the given string
        function obj = setPriorProbabilities( obj )
            
            for i = 1:obj.hypothesesCount
                
                currentString = obj.getHypothesisString( i-1 );
                
                datasetH = ...
                    obj.trainingDataset.getDatabaseForHypothesis(...
                    currentString );

                obj.p(i) = obj.estimateProbabilityUsingML( datasetH );

            
            end % for i
            
        end % function setPriorProbabilities
        
        % Determine which categories are associated with the given
        % hypothesis
        function hString = getHypothesisString( obj, hypothesisIndex )
                
            hString = [];
            categoriesPerHypothesis = ...
            size( obj.hypothesesStrings, 1 ) / ...
            obj.hypothesesCount;   
        
            for i = 1:categoriesPerHypothesis
                
                hString = [hString; obj.hypothesesStrings(...
                    hypothesisIndex * categoriesPerHypothesis + i, : )];
                
            end % for i
                
        end % function getHypothesisString   
        
        % Compute the log likelihood ration assuming a bernoulli model
        function [ratio, numeratorRunningSum] = ...
                computeLogLikelihoodRatioUsingBernoulli(...
                obj, singleEntryDataset )
            
            ratio = 0;
            
            % The first row will remain always zero
            numeratorRunningSum = zeros( 1, obj.hypothesesCount );
            
            for i = 1:singleEntryDataset.getFeaturesCount()
                
                b = singleEntryDataset.isFeaturePresent( i );
                
                for j = 1:obj.hypothesesCount-1
                
                    numerator = ...
                        ( obj.getConditionalProb( i, j ) ) ^ b * ...
                        ( 1 - obj.getConditionalProb( i, j ) ) ^ ( 1 - b );

                    numeratorRunningSum( j+1 ) = ...
                        numeratorRunningSum( j+1 ) + ...
                        log( numerator );
                
                end % for j
                
                denominator = ...
                    ( obj.getConditionalProb( i, 0 ) ) ^ b * ...
                    ( 1 - obj.getConditionalProb( i, 0 ) ) ^ ( 1 - b );
                
                numeratorRunningSum( 1 ) = ...
                    numeratorRunningSum( 1 ) + log( denominator );
                
                ratio = ratio + log( numerator / denominator );
                
            end % for i
            
        end % function computeLogLikelihoodRatioUsingBernoulli    
                
        % A wrapper function to get access to conditional probabilities
        % This makes things a bit more straightforward as we can 
        % use index 0 to get access to probabilities given H0
        function prob = getConditionalProb( obj, feature, hypothesis )
            
            prob = obj.conditionalProbabilities( feature, hypothesis+1 );
            
        end % function getConditionalProb
        
        % Compute the overall accuracy defined
        % as the number of corrctly classified
        % samples to total number of samples
        function obj = computeAccuracy( obj )
            
            testDataset = obj.dataset.getTestDatabase();
            trueResults = testDataset.getHypothesis( obj.hypothesesCount );
            compare = ( trueResults == obj.assignedClasses );
            obj.accuracy = sum( compare ) / length( compare );
            
        end % end class computeAccuracy
        
        % Compute the conditional probability using the multinomial
        % model
        function estimatedProbability = ...
                estimateConditionalProbUsingMultinomialModel(...
                obj, i, hypothesisDataset, Nj )
            
            tfij = hypothesisDataset.getFeatureFrequency( i );   
            estimatedProbability = ( tfij + 1 ) / ...
                ( Nj + hypothesisDataset.getFeaturesCount() );
            
        end % end estimateConditionalProbUsingMultinomialModel
        
    end % class utility functions
    
    methods( Access = public )
        
        % Set the costs as give
        % Costs must be a vector in this format:
        % (For binary case)
        % [C00, C01, C10, C11]        
        function obj = setCosts( obj, newCosts )

            obj.costs = newCosts;

        end % function setCost
    
        % Learn the conditinal probabilities assuming a Bernoulli model
        function obj = trainUsingBernoulliModel( obj )            
            
            obj = obj.setPriorProbabilities();
            
            for h = 1:obj.hypothesesCount
                
                datasetOfCurrentHypothesis = ...
                    obj.trainingDataset.getDatabaseForHypothesis(...
                    obj.getHypothesisString( h-1 ) );
              
                % Exctract and save all conditional probabilities for 
                % feature i given hypothesis
                for i = 1:obj.dataset.getFeaturesCount()

                    probability = ...
                        obj.estimateConditionalProbUsingBernoulliModel(...
                        i, datasetOfCurrentHypothesis );

                    obj.conditionalProbabilities( i, h ) = probability;

                end % for i

            end % for h         
            
        end % function trainUsingBernoulliModel 
        
        % Compute the conditional probabilities using the 
        % multinomial model and the training dataset
        function obj = trainUsingMultinomialModel( obj )
            
            obj = obj.setPriorProbabilities();
            
            for h = 1:obj.hypothesesCount
                
                datasetOfCurrentHypothesis = ...
                    obj.trainingDataset.getDatabaseForHypothesis(...
                    obj.getHypothesisString( h-1 ) );
                
                N = datasetOfCurrentHypothesis.getAllTermsFrequencies();
              
                % Exctract and save all conditional probabilities for 
                % feature i given hypothesis
                for i = 1:obj.dataset.getFeaturesCount()

                    probability = ...
                    obj.estimateConditionalProbUsingMultinomialModel(...
                        i, datasetOfCurrentHypothesis, N );

                    obj.conditionalProbabilities( i, h ) = probability;

                end % for i

            end % for h          
            
        end % function trainUsingMultinomialModel
        
        % Print results
        % Mostly for debugging purposes
        function printResults( obj )
            
            fprintf( 'Acc = %0.4f (%%)\n', obj.accuracy * 100 );
            fprintf( 'P_f = %0.4f\n', obj.falseAlarmProb );
            fprintf( 'P_d = %0.4f\n', obj.detectionProb );
            fprintf( 'P_m = %0.4f\n', obj.missDetectionProb );
            fprintf( 'P_e = %0.4f\n', obj.errorProb );
            
        end % function printResults
        
    end % class public services
    
end % class GenericTest