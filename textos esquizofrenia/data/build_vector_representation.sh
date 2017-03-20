#!/bin/bash


function preprocess_training_testing_sets {
	(cd ./training;for i in $( ls Experimental/*.ana ); do python ../../src/postagging.py -i $i; done;)
	(cd ./training;for i in $( ls Control/*.ana ); do python ../../src/postagging.py -i $i; done;)

	(cd ./training;cat tags_found |sort |uniq > tags)
	(cd ./training;python ../../src/pos_feature.py)


	(cd ./training;for i in $( ls Control/*.txt );do echo $i 1;done > classes)
	(cd ./training;for i in $( ls Experimental/*.txt );do echo $i -1;done >> classes)

	(cd ./training;sed -ie 's/Control\///g' classes)
	(cd ./training;sed -ie 's/Experimental\///g' classes)
	(cd ./training;rm classese)



	(cd ./testing;for i in $( ls Experimental/*.ana ); do python ../../src/postagging.py -i $i; done;)
	(cd ./testing;for i in $( ls Control/*.ana ); do python ../../src/postagging.py -i $i; done;)

	(cd ./testing;cat tags_found |sort |uniq > tags)
	(cd ./testing;python ../../src/pos_feature.py)


	(cd ./testing;for i in $( ls Control/*.txt );do echo $i 1;done > classes)
	(cd ./testing;for i in $( ls Experimental/*.txt );do echo $i -1;done >> classes)
	(cd ./testing;sed -ie 's/Control\///g' classes)
	(cd ./testing;sed -ie 's/Experimental\///g' classes)
	(cd ./testing;rm classese)
}

function build_vectors_for_training_testing_sets {
	# receives the freeling feature codes separated by comma: i.e. N,P,S
	(cd ./training;python ../../src/vector_representation.py $1)
	(cd ./training;cp features ../testing/)
	(cd ./testing;python ../../src/vector_representation.py use-features)
}


if [ $# -eq 0 ]
  then
    echo "Usage:$0 freeling_features_to_use"
    exit 1
fi

preprocess_training_testing_sets
build_vectors_for_training_testing_sets $1 
