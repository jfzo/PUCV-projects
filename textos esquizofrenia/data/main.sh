#!/bin/bash



function create_links {
	rm -f training testing
	ln -s $1 training
	ln -s $2 testing
}

function execute_experimentation {
	# A,C,D,F,I,N,P,R,S,V,W,Z
	rm -f testing/results_$1.log
	./build_vector_representation.sh $1
	python ../src/classify.py --tr training --ts testing -o testing -r 10 -k 3 -l testing/results_$1.log
}


features='A,C,D,F,I,N,P,R,S,V,W,Z'

if [ $# -eq 0 ]
  then
    echo "Usage:$0 features_to_use"
else
    features=$1
fi

echo "Features to be used: "$features


# Generate each dataset
create_links "ab" Relato_C
execute_experimentation $features

create_links "ac" Relato_B
execute_experimentation $features

create_links "bc" Relato_A
execute_experimentation $features

# for each one and for each tuple of features execute: ./build_vector_representation.sh comma_separated_tuple_of_pos_tags



