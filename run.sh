dataset="mag"
path="pr"
architecture="cross"
test_file="original"

python3 main.py --bert_model scibert_scivocab_uncased/ --output_dir ${dataset}_output_${path}/ --train_dir ${dataset}_${path}/ --test_file ${dataset}_test/test_${test_file}.txt --use_pretrain --architecture ${architecture}
python3 main.py --bert_model scibert_scivocab_uncased/ --output_dir ${dataset}_output_${path}/ --train_dir ${dataset}_${path}/ --test_file ${dataset}_test/test_${test_file}.txt --use_pretrain --architecture ${architecture} --eval

python3 calcsim.py --output_dir ${dataset}_output_${path}/ --architecture ${architecture}
python3 patk.py --output_dir ${dataset}_output_${path}/ --architecture ${architecture}
