export TRAIN_FILE=~/PycharmProjects/Crisis/Datasets/COVID/01.jsonl.en.txt
export TEST_FILE=~/PycharmProjects/Crisis/Datasets/Tagged/10k.jsonl.en.txt

python /home/stowe/GitHub/transformers/examples/language-modeling/run_language_modeling.py \
       --output_dir=covid-test \
       --model_type=bert \
       --model_name_or_path=bert-base-cased \
       --do_train \
       --train_data_file=$TRAIN_FILE \
       --do_eval \
       --eval_data_file=$TEST_FILE \
           --mlm
