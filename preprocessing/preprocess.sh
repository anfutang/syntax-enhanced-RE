#!/bin/sh

DATASET_NAME=$1
INPUT_DIR=$2

# generate base files
printf "######## copying base files...\n"

ip=${INPUT_DIR}/wordpiece_level_files/${DATASET_NAME}
op=${INPUT_DIR}/base_files
if ! test -d $op; then mkdir $op; fi

op=${INPUT_DIR}/base_files/${DATASET_NAME}
if ! test -d $op; then mkdir $op; fi

python3 generate_base_files.py --input_dir $ip --output_dir $op > $op/log.txt

# remove entity markers
printf "######## removing entity markers...\n"

op=${INPUT_DIR}/word_level_files
if ! test -d $op; then mkdir $op; fi

op=${INPUT_DIR}/word_level_files/${DATASET_NAME}
if ! test -d $op; then mkdir $op; fi

for fn in train dev test
do
  python3 remove_markers.py --data_fn $ip/${fn}.pkl --output_dir $op --output_fn ${fn}.pkl --error_output_fn ${fn}_error.pkl > ${op}/log_${fn}.txt
done

# dependency parses

printf "######## dependency parsing...\n"

ip=${INPUT_DIR}/word_level_files/${DATASET_NAME}

op=${INPUT_DIR}/dependency_parses
if ! test -d $op; then mkdir $op; fi

op=${INPUT_DIR}/dependency_parses/${DATASET_NAME}
if ! test -d $op; then mkdir $op; fi

for fn in train dev test
do 
  python3 dependency_parse.py --data_fn $ip/${fn}.pkl --output_dir $op --output_fn ${fn}.pkl > ${op}/log_${fn}.txt
done

# constituency parses

printf "######## constituency parsing...\n"

op=${INPUT_DIR}/constituency_parses
if ! test -d $op; then mkdir $op; fi

op=${INPUT_DIR}/constituency_parses/${DATASET_NAME}
if ! test -d $op; then mkdir $op; fi

for fn in train dev test
do
  python3 constituency_parse.py --data_fn ${INPUT_DIR}/word_level_files/${DATASET_NAME}/${fn}.pkl --dataset_name ${fn} --output_dir $op > ${op}/log_${fn}.txt
done

# generate wordpiece-to-const spans and linearized constituency trees

wp_ip=${INPUT_DIR}/wordpiece_level_files/${DATASET_NAME}
word_ip=${INPUT_DIR}/word_level_files/${DATASET_NAME}
const_ip=${INPUT_DIR}/constituency_parses/${DATASET_NAME}

op=${INPUT_DIR}/constituency_files
if ! test -d $op; then mkdir $op; fi

op=${INPUT_DIR}/constituency_files/${DATASET_NAME}
if ! test -d $op; then mkdir $op; fi

printf "######## generating constituency files for CE and CT models."

python3 generate_constituency_files.py --wp_file_dir $wp_ip --word_file_dir $word_ip --const_parse_dir $const_ip --output_dir $op > ${op}/log.txt

#printf "filtering linearized constituency trees...\n"
#python3 filter_constituency_seqs.py --data_dir ${op}

# generate adjancency matrices (for late-fusion)
dep_ip=${INPUT_DIR}/dependency_parses/${DATASET_NAME}
op=${INPUT_DIR}/dependency_files
if ! test -d $op; then mkdir $op; fi

op=${INPUT_DIR}/dependency_files/${DATASET_NAME}
if ! test -d $op; then mkdir $op; fi

printf "######## generating adjacency matrices...\n"

python3 generate_adj_matrices.py --wp_file_dir $wp_ip --word_file_dir $word_ip --dep_parse_dir $dep_ip --output_dir $op > ${op}/log_adj.txt

# generate pair-wise syntactic distance matrices and syntactic depth lists

printf "######## generating distance matrices and depths (syntactic probe)...\n"

python3 generate_dependency_dists_and_depth.py --wp_file_dir $wp_ip --word_file_dir $word_ip --dep_parse_dir $dep_ip --output_dir $op > ${op}/log_dist_depth.txt

# turn syntactic distances and depths into categories (regression to classification)

printf "######## categorizing syntactic distances and depths ...\n"

python3 categorize_dists_depths.py --dep_file_dir $op > ${op}/log_categorization.txt

printf "######## exit.\n"

