clc;
fprintf( "Be patient! The execution might take multiple minutes.\n" );
fprintf( "\n-------- Binary Test Based on Bernoulli Model ---------\n" );
clear;
filename = 'SeeClickFix_AlbanyCounty_February_2018.csv';

% Cost order:
% [c00, c01, c10, c11]
costs = [0, 1, 1, 0];

dataset = Dataset( 'filename', filename );
classifier = BinaryTest( dataset, 2 );
classifier = classifier.trainUsingBernoulliModel();
classifier = classifier.setCosts( costs );
classifier = classifier.test( 'Bernouli' );
classifier = classifier.analyzeResults();
classifier.printResults();
%classifier.computeROC( 'Bernouli' );

fprintf( "\n------- Binary Test Based on Multinomial Model --------\n" );
clear;
filename = 'SeeClickFix_AlbanyCounty_February_2018.csv';
costs = [0, 1, 1, 0];

dataset = Dataset( 'filename', filename );
classifier = BinaryTest( dataset, 2 );
classifier = classifier.trainUsingMultinomialModel();
classifier = classifier.setCosts( costs );
classifier = classifier.test(  'Multinomial' );
classifier = classifier.analyzeResults();
classifier.printResults();
%classifier.computeROC( 'Multinomial' );

fprintf( "\n------------ M Test Based on Bernoulli Model -----------\n" );
clear;
filename = 'SeeClickFix_AlbanyCounty_February_2018.csv';

% Costs order:
% [c00, c01, c02 ..., c10, c11, c12,... ]
costs = [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0];

dataset = Dataset( 'filename', filename );
classifier = MTest( dataset, 4 );
classifier = classifier.setCosts( costs );
classifier = classifier.trainUsingBernoulliModel();
classifier = classifier.test( 'Bernouli' );
classifier = classifier.analyzeResults();
classifier.printResults();


fprintf( "\n--------- M Test Based on Multinomial Model ----------\n" );
clear;
filename = 'SeeClickFix_AlbanyCounty_February_2018.csv';
costs = [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0];

dataset = Dataset( 'filename', filename );
classifier = MTest( dataset, 4 );
classifier = classifier.setCosts( costs );
classifier = classifier.trainUsingMultinomialModel();
classifier = classifier.test( 'Multinomial' );
classifier = classifier.analyzeResults();
classifier.printResults();