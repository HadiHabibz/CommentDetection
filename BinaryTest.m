classdef BinaryTest < GenericTest
    
    properties( Access = private )
               
        thresholdFlag;
        modelName;
        
    end % private class memebers
    
    methods( Access = public )
        
        function obj = BinaryTest( completeDataset, HCount )
            
            obj@GenericTest( completeDataset, HCount ); 
                        
            % Default costs minimize the probability of error
            obj = obj.setCosts( [1, 0, 1, 0] );

            obj.thresholdFlag = 'auto';
                        
        end % constructor
                
        % Test the data in the test dataset one by one
        % Costs must be a vector in this format:
        % [C10, C00, C01, C11]
        function [obj, logRatio] = test( obj, model )
            
            testDataset = obj.dataset.getTestDatabase();
            logRatio = zeros( 1, testDataset.getEntriesCount() );
            obj.modelName = model;
            
            for entry = 1:testDataset.getEntriesCount()
                
                currentEntry = testDataset.getEntry( entry );
                
                if contains( lower( model ), 'bern' )
                    logRatio( entry ) = ...
                        obj.computeLogLikelihoodRatioUsingBernoulli(...
                        currentEntry );
                    
                else
                    logRatio( entry ) = ...
                        obj.computeLogLikelihoodRatioUsingMultinomial(...
                        currentEntry );
                     
                end % end if...else
                
            end % for entry
            
            obj.assignedClasses = obj.classify( logRatio, 0 );
            
        end % function test
        
        % Compute the ROC 
        function computeROC( obj, model )

            [obj, logRatios] = obj.test( model );
            
            threshold = [...
                1e-8:10*1e-8:1e-4;...
                1e-4:10*1e-4:1;
                1:10:1e4;
                1e4:10*1e4:1e8;...
                1e8:10*1e8:1e12];
            
            pd = zeros( size( threshold ) );
            pf = zeros( size( threshold ) );
            
            for i = 1:size( threshold, 1 )

                for j = 1:size( threshold, 2 )
                    
                    manualThreshold = threshold( i, j );
                    obj.thresholdFlag = 'manual';
                    obj.assignedClasses = obj.classify(...
                    logRatios, manualThreshold );
                    obj = obj.analyzeResults();
                    pd( i, j ) = obj.detectionProb;
                    pf( i, j ) = obj.falseAlarmProb;
                    
                end % for j
   
            end % for i
            
            pf = pf';
            pd = pd';
            
            figure;
            plot( pf( : ), pd( : ), 'LineWidth', 3 );
            xlabel( 'P_F' );
            ylabel( 'P_D' );
            title( obj.modelName );
            grid ON;
            
        end % function computeROC
        
        function obj = analyzeResults( obj )
            
            obj = obj.computeAccuracy();
            obj = obj.computeFalseAlarmProbability();
            obj = obj.computeDetectionProbability();
            obj = obj.computeErrorProbability();
            
        end % function analyzeResults
        
        % Compute the threshold
        function threshold = getThreshold( obj, mannualThreshold )
            
            if contains( obj.thresholdFlag, 'auto' )
                
                threshold = ( obj.p(1) / obj.p(2) ) *...
                    ( ( obj.costs( 3 ) - obj.costs( 1 ) ) ...
                    / ( obj.costs( 2 ) - obj.costs( 4 ) ) );
                
                return;

            end % if
            
            threshold = mannualThreshold;
            
        end % function getThreshold
        
        % Assigned a category to each entry
        function assignedClasses = classify(...
                obj, logRatio, mannualThreshold  )
            
            assignedClasses = ( logRatio >=...
                log( obj.getThreshold( mannualThreshold ) ) );
            
        end % function classify
        
        % Compute the probability of error
        function obj = computeErrorProbability( obj )
            
            obj.errorProb = obj.p(1) * obj.falseAlarmProb + ...
                obj.p(2) * obj.missDetectionProb;
            
        end % function computeErrorProbability 
        
        % Compute detection probability
        function obj = computeDetectionProbability( obj )
            
            testDataset = obj.dataset.getTestDatabase();
            trueResults = testDataset.getHypothesis( obj.hypothesesCount ); 
            
            % Pick the indices of all element that belong
            % to hypothesis 1
            indicesH1 = find( trueResults == 1 );
            
            % Pick the indices of all elements that are
            % classified as hypothesis 1
            indicesClassifiedAsH1 = find( obj.assignedClasses == 1 );
            
            detectionCount = length( intersect(...
                indicesH1, indicesClassifiedAsH1 ) );
            
            obj.detectionProb = detectionCount / ...
                length( indicesH1 );      
            
            obj.missDetectionProb = 1 - obj.detectionProb;
            
        end % function computeDetectionProbability
        
        % Compute the probability of false alarm
        function obj = computeFalseAlarmProbability( obj )
           
            testDataset = obj.dataset.getTestDatabase();
            trueResults = testDataset.getHypothesis( obj.hypothesesCount );
            
            % Pick the indices of all element that belong
            % to hypothesis 0
            indicesH0 = find( trueResults == 0 );
            
            % Pick the indices of all elements that are
            % classified as hypothesis 1
            indicesClassifiedAsH1 = find( obj.assignedClasses == 1 );
            
            % Compute the number of elements that actually belong to
            % H0 but are classified as H1
            falseAlarmCount = length( intersect(...
                indicesH0, indicesClassifiedAsH1   ) );
            
            obj.falseAlarmProb = falseAlarmCount /...
                length( indicesH0 );            
            
        end % function computeFalseAlarmProbability
                
        % Compute the log likelihood ration assuming a bernoulli model
        function ratio = computeLogLikelihoodRatioUsingMultinomial(...
                obj, singleEntryDataset )
            
            ratio = 0;
            
            for i = 1:singleEntryDataset.getFeaturesCount()
                
                b = singleEntryDataset.getFeatureValue( i );
                
                currentRatio = obj.getConditionalProb( i, 1 ) / ...
                    obj.getConditionalProb( i, 0 );
                
                ratio = ratio + b * log( currentRatio );
                
            end % for i
            
        end % function computeLogLikelihoodRatioUsingMultinomial
        
    end % class services
    
end % class BinaryTest