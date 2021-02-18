classdef MTest < GenericTest
    
    properties( Access = private )
        
    end % class data memebers
    
    methods( Access = public )
        
        % Constructor
        function obj = MTest( completeDataset, HCount )
           
            obj@GenericTest( completeDataset, HCount );
            
            obj.hypothesesCount = 4;      
            obj.p = zeros( 1, obj.hypothesesCount );
            
        end % Constructor
        
        % Test the data in the test dataset one by one
        % Costs must be a vector in this format:
        % [C00, C01, C02, ..., C10, C11, ...]
        function obj = test( obj, model )
            
            testDataset = obj.dataset.getTestDatabase();
            beta = zeros( testDataset.getEntriesCount(),...
                obj.hypothesesCount );
            
            for entry = 1:testDataset.getEntriesCount()
                
                currentEntry = testDataset.getEntry( entry );
                
                if contains( lower( model ), 'bern' )
                    
                    for i = 1:obj.hypothesesCount
                        
                        beta( entry, i ) = ...
                            obj.getBeta_i( i-1, currentEntry );
                        
                    end % for i
                    
                else
                    for i = 1:obj.hypothesesCount
                        
                        beta( entry, i ) = ...
                            obj.getBeta_i( i-1, currentEntry );
                        
                    end % for i
                     
                end % end if...else
                
            end % for entry
            
            obj.assignedClasses = obj.classify( beta );
            
        end % function test
        
        % Assigned a category to each entry
        function assignedClasses = classify( obj, beta  )
            
            assignedClasses = zeros( 1, size( beta, 1 ) ); 
            
            for i = 1:size( beta, 1 )
                
                [~, assignedClasses( i )] = min( beta( i, : ) );
                
            end % for i
            
            assignedClasses = assignedClasses - 1;
            
        end % function classify
        
        
        % Compute the beta value for hypothesis i
        function beta = getBeta_i( obj, i, currentEntry )
            
            [~, conditionalProbabilities]= ...
                obj.computeLogLikelihoodRatioUsingBernoulli(...
                currentEntry );
            
            conditionalProbabilities = exp( conditionalProbabilities );
            
            conditionalProbabilities = ...
                conditionalProbabilities .* obj.p;
            
            conditionalProbabilities = ...
                conditionalProbabilities .*...
                obj.costs( i * obj.hypothesesCount + 1:...
                ( i + 1 ) * obj.hypothesesCount );
            
            beta = sum( conditionalProbabilities );
                    
        end % function getBeta_i
        
        function obj = analyzeResults( obj )
            
            obj = obj.computeAccuracy();
            [obj, pf] = obj.computeFalseAlarmProbability();
            [obj, pd] = obj.computeDetectionProbability();
            obj = obj.computeErrorProbability( pf, pd );
            
        end % function analyzeResults
        
        % Compute the probability of error
        function obj = computeErrorProbability( obj, pf, pd )
            
            obj.errorProb = 0;
            
            % Compute probability of error for each hypothesis
            % Add them up to get the final error probability
            for h = 1:obj.hypothesesCount
                
                % Group all other hypotheses into one null
                % hypothesis
                indexOfNoneSelectedHypotheses = ...
                    setdiff( ( 1:obj.hypothesesCount ), h );
                
                pe = ( 1 - obj.p( h ) ) * ...
                    pf( h ) + obj.p( h ) * ( 1 - pd( h ) );
                
                obj.errorProb = obj.errorProb + pe;
                
            end % for h
                        
        end % function computeErrorProbability 
        
        % Compute the probability of false alarm
        function [obj, pf] = computeFalseAlarmProbability( obj )
           
            testDataset = obj.dataset.getTestDatabase();
            trueResults = testDataset.getHypothesis( obj.hypothesesCount );
            allHypotheses = 0:1:obj.hypothesesCount-1;
            obj.falseAlarmProb = 0;
            pf = zeros( 1, obj.hypothesesCount );
            
            for i = 0:1:obj.hypothesesCount-1
                
                % All other hypotheses but the current one
                % We group all of them together to make 
                % everything binary again.
                wrongHypotheses = setdiff( allHypotheses, i );
                
                % Pick the indices of all element that belong
                % to hypothesis i
                indicesH = find( obj.assignedClasses == i );
                indicesClassifiedAsH = [];
                
                for j = 1:length( wrongHypotheses )
                    
                    % Pick the indices of all elements that are
                    % classified as any hypotheses other than i
                    indicesClassifiedAsH  =...
                        [indicesClassifiedAsH, find(...
                        trueResults == wrongHypotheses( j ) )];
                    
                end % for j
                
                % Compute the number of elements that actually belong to
                % H0 but are classified as H1
                falseAlarmCount = length( intersect(...
                indicesH, indicesClassifiedAsH ) );
            
                % Save these values.
                % We will need them for computing probability of error
                pf( i+1 ) = falseAlarmCount / length( indicesH );
                
                obj.falseAlarmProb = obj.falseAlarmProb + ...
                    obj.p( i+1 ) * pf( i + 1 );
                
            end % for i      
            
        end % function computeFalseAlarmProbability
        
        % Compute detection probability
        % This way, detection probability is the same
        % As accuracy
        function [obj, pd] = computeDetectionProbability( obj )
            
            testDataset = obj.dataset.getTestDatabase();
            trueResults = testDataset.getHypothesis( obj.hypothesesCount ); 
            obj.detectionProb = 0;
            pd = zeros( 1, obj.hypothesesCount );
            
            for i = 0:1:obj.hypothesesCount-1
                
                % Pick the indices of all element that belong
                % to hypothesis i
                indicesH = find( trueResults == i );
                
                % Pick the indices of all elements that are
                % classified as as i, whether correctly or not.
                indicesClassifiedAsH  = find(...
                    obj.assignedClasses == i );
                
                % Get all those samples that are correctly detected
                correctlyDetectedSamples = intersect(...
                    indicesH, indicesClassifiedAsH );
                
                % Save the detection probability
                % We will need this for computing error probabilities
                pd( i + 1 ) = length( correctlyDetectedSamples ) / ...
                    length( indicesH );
                
                % Compute the detection probability for this 
                % hypothesis
                obj.detectionProb = obj.detectionProb + ...
                    pd( i+1 ) * obj.p( i+1 ); 
                
            end % for i
            
            obj.missDetectionProb = 1 - obj.detectionProb;

        end % function computeDetectionProbability
        
    end % class service functions
    
end % class MTest