classdef Dataset
    
    properties( Access = private )
        
        % Store the entire data in a matrix
        dataMatrix;
        
        % Store the label associated with each entry
        labels;
        
        % 1 indicates test
        % 0 indicates train
        testOrTrainLabel;
        
        
    end % properties
    
    methods( Access = public )
        
        % Constructor function
        function obj = Dataset( varargin )
            
            [filename, dataToCopy, labels] = ...
                obj.parseConstructorsInput( varargin );
            
            if ~contains( filename, 'none' )
             
                obj = obj.loadCSVFile( filename );
                obj = obj.splitTestAndTrainingData();
                return;
                
            end % if
            
            obj.dataMatrix = dataToCopy;
            obj.labels = labels;
            
        end % constructor
        
        % Basically, determine what inputs are given
        % to the constructor.
        function [filename, dataMatrix, labels] =...
                parseConstructorsInput( obj, inputs )
            
            dataMatrix = 0; 
            labels = 0;
            filename = 'none';
            
            for i = 1:2:length( inputs )
            
                string = cell2mat( inputs( i ) );

                if contains( lower( string ), 'filename' )
                    filename = cell2mat( inputs( i+1 ) );
                    
                elseif contains( lower( string ), 'datamatrix' )
                    dataMatrix = cell2mat( inputs( i+1 ) );
                    
                elseif contains( lower( string ), 'labels' )
                    labels = cell2mat( inputs( i+ 1 ) );
                    
                end % if

            end % for i
            
        end % function parseConstructorsInput
        
        % Load data. Read it first as table to cope
        % with non numerical entries
        function obj = loadCSVFile( obj, filename )
            
            myTable = readtable( filename );
            obj.dataMatrix = table2array( myTable( 1:end, 1:end-1 ) );
            fullLables = table2array( myTable( 1:end, end ) );
            obj = obj.loadTruncatedLabels( fullLables );
            
        end % function loadCSVFile
        
        % Return all entries that are used for training
        % Regradless of their category
        function trainingDatabase = getTrainingDatabase( obj )
            
            indices = find( ( obj.testOrTrainLabel == 0 ) );
            trainData = obj.dataMatrix( indices, : );
            newLables = obj.labels( indices, : );
            trainingDatabase = Dataset( 'datamatrix', trainData, ...
                'labels', newLables );
            
        end % function getTrainingDatabase
        
        % Return all entries that are used for test
        % Regradless of their category
        function testDatabase = getTestDatabase( obj )
            
            indices = find( ( obj.testOrTrainLabel == 1 ) );
            testData = obj.dataMatrix( indices, : );
            newLabels = obj.labels( indices, : );
            testDatabase = Dataset( 'datamatrix', testData, ...
                'labels', newLabels );
            
        end % function getTestDatabase
        
        % Exctract all entries that pertain to the given hypothesis
        function hypothesisDatabase = ...
                getDatabaseForHypothesis( obj, categories )
            
            indices = [];
            
            % Find the indices of the entries
            for i = 1:size( obj.labels, 1 )
                
                for j = 1:size( categories, 1 )
                    
                    if contains( obj.labels( i, : ),...
                            categories( j, : ) )
                        
                        indices = [indices, i];
                        break;

                    end % if
                
                end % for j
                
            end % for i
            
            newData = obj.dataMatrix( indices, : );
            newLabels = obj.labels( indices, : );
            hypothesisDatabase = Dataset( 'datamatrix', newData, ...
                'labels', newLabels );
            
        end % function hyptothesisDatabase
        
        % First determine how many entries there are for each
        % category. For each category, assign the first half
        % to test dataset and the second half to train dataset
        function obj = splitTestAndTrainingData( obj )
            
            categories = [ 'Sign'; 'Traf'; 'Park'; 'Code' ];
            counts = zeros( 1, length( categories ) );
            
            for cat = 1:length( categories )
                
                counts( cat ) = ...
                    obj.getCategoryCount( categories( cat, : ) );
                
            end % for cat
            
            counts = floor( counts / 2 );
            obj.testOrTrainLabel = zeros( 1, length( obj.labels ) );
            
            for i = 1:size( obj.dataMatrix, 1 )
                
                catNumber = obj.getCategoryNumber( i );
                counts( catNumber ) = counts( catNumber ) - 1;
                
                if counts( catNumber ) >= 0
                    obj.testOrTrainLabel( i ) = 1;
                    
                else
                    obj.testOrTrainLabel( i ) = 0;
                    
                end % if
                
            end % for dataEntry
            
        end % function splitTestAndTrainingData
        
        % Map string categories to integer
        % Stick with this nymbers
        function categoryNumber = getCategoryNumber( obj, entry )
            
            category = obj.labels( entry, : );
            
            if contains( category, 'Sign' )
                categoryNumber = 1;
                
            elseif contains( category, 'Traf' )
                 categoryNumber = 2;
                
            elseif contains( category, 'Park' )
                 categoryNumber = 3;
                
            else
                 categoryNumber = 4;
                
            end % if
            
        end % function getCategoryNumber
        
        % Basically, we only use the first four letters
        % to determine the label. This helps us to save
        % strings in a matrix
        function obj = loadTruncatedLabels( obj, fullLabels )
            
            obj.labels = [];
            
            for i = 1:length( fullLabels )
                
                string = cell2mat( fullLabels( i ) );
                obj.labels = [obj.labels; string( 1:4 )];
                
            end % for i
            
        end % function loadTruncatedLabels
        
        % Check how many entries there are that are associated
        % with the given category. We need this for spliting our
        % dataset
        function categoryCount = getCategoryCount( obj, category )
            
            categoryCount = 0;
            category = category( 1:4 );
            
            for i = 1:length( obj.labels )
                
                categoryCount = categoryCount +...
                    ( contains( obj.labels( i, : ), category ) );
                
            end % for
            
        end % function getCategoryCount
        
        % Get number of entries corresponding to the 
        % number of rows in data 
        function entriesCount = getEntriesCount( obj )
            
            entriesCount = size( obj.labels, 1 );
            
        end % function getEntriesCount
        
        % Return the number of entries for which feature number
        % 'featureIndex' is nonzero. Feature number starts from 
        % 1, so discard the first column that contains ids.
        function count = getEntriesCountForTheFeature( obj, featureIndex )
            
            features = obj.dataMatrix( :, featureIndex + 1 );
            indices = find( features ~= 0 );            
            count = length( indices );
            
        end % end getEntriesCountForTheFeature
        
        % Get the total number of features
        function count = getFeaturesCount( obj )
            
            % The first column is the IDS
            % The last one is the categories
            % Skip these two
            count = size( obj.dataMatrix, 2 ) - 2;
            
        end % function getFeaturesCount
        
        % Get a dataset containing only one entry
        function singleEntryDataset = getEntry( obj, entry )
            
            newData = obj.dataMatrix( entry, : );
            newLabels = obj.labels( entry, : );
            singleEntryDataset = Dataset( 'datamatrix', newData, ...
                'labels', newLabels );
            
        end % function getEntry
        
        % Given the feature index, check if the value of that
        % feature is zero or not. Return zero if it is zero.
        % Otherwise, return 1. Works only for single entry
        % dataset
        function yesOrNo = isFeaturePresent( obj, featureIndex )
            
            % Skip the first column, which is the entry IDs. 
            yesOrNo = ( obj.dataMatrix( 1, featureIndex + 1 ) ~= 0 );
            
        end % function isFeaturePresent
        
        % Determine which hypothesis the entries belong to
        % 'Sign' and 'Traf' are in hypothesis 1 while
        % 'Park' and 'Code' are in hypothesis 0.
        function hypothesis = getHypothesis( obj, hCount )
            
            hypothesis = zeros( 1, obj.getEntriesCount() );
            
            if hCount == 2
                
                for i = 1:obj.getEntriesCount()

                    categoryNumber = obj.getCategoryNumber( i );

                    % 3 and 4 correspond to 'Park' and 'Code', respectively
                    if categoryNumber == 3 || categoryNumber == 4
                        hypothesis( i ) = 0;

                    else
                        hypothesis( i ) = 1;

                    end % if

                end % for i
                
            else
                
                for i = 1:obj.getEntriesCount()

                    categoryNumber = obj.getCategoryNumber( i );

                    % 3 and 4 correspond to 'Park' and 'Code', respectively
                    if categoryNumber == 3 
                        hypothesis( i ) = 0;

                    elseif categoryNumber == 4
                        hypothesis( i ) = 1;
                        
                    elseif categoryNumber == 1
                        hypothesis( i ) = 2;
                        
                    else
                        hypothesis( i ) = 3;

                    end % if

                end % for i
                
            end % if
            
        end % function getHypothesis
        
        % Return the frequency of each feature accross all entries
        function featureFrequency = ...
                getFeatureFrequency( obj, featureIndex )
            
            % Skip the first column. It is inde
            featureExtracted = obj.dataMatrix( :, featureIndex+1 );
            featureFrequency = sum( featureExtracted );
            
        end % function getFeatureFrequency
        
        % Get the total frequency of all words
        function frequency = getAllTermsFrequencies( obj )
            
            frequency = 0;
            
            for i = 1:obj.getFeaturesCount
                
                frequency = frequency + obj.getFeatureFrequency( i );
                
            end % for i
            
        end % function getAllTermsFrequencies
        
        % Return how many times the given feature is repeated 
        % in the dataset. The database must be single entry only.
        function featureValue = getFeatureValue( obj, featureIndex )
            
            featureValue = obj.dataMatrix( 1, featureIndex + 1 );
            
        end % function getFeatureValue
        
    end % public methods
    
end % class Dataset 