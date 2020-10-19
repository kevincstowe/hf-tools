for count in 0 1 2 3 4 5 6 7 8 9
do
	export TRAIN_FILE="/home/stowe/PycharmProjects/Crisis/Datasets/Drezde/${count}.txt"
	export TEST_FILE=/home/stowe/PycharmProjects/Crisis/Datasets/Drezde/0.txt

	echo "Running file number $count"
	python3 /home/stowe/GitHub/adapter-transformers/examples/language-modeling/run_language_modeling.py \
       	--output_dir="0${count}-drezde-adapter" \
       	--model_type=bert \
       	--model_name_or_path=bert-base-cased \
       	--do_train \
       	--train_data_file=$TRAIN_FILE \
       	--do_eval \
       	--eval_data_file=$TEST_FILE \
           --mlm \
       	--overwrite_output_dir \
       	--train_adapter \
       	--language=twitter_hurricane
done

