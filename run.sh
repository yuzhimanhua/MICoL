dataset=MAG
architecture=cross

python3 main.py --bert_model scibert_scivocab_uncased/ \
                --train_dir ${dataset}_input/ --test_file ${dataset}_input/test.txt \
                --output_dir ${dataset}_output/ --architecture ${architecture} --use_pretrain
python3 main.py --bert_model scibert_scivocab_uncased/ \
                --train_dir ${dataset}_input/ --test_file ${dataset}_input/test.txt \
                --output_dir ${dataset}_output/ --architecture ${architecture} --use_pretrain --eval

python3 postprocess.py --output_dir ${dataset}_output/ --architecture ${architecture}
python3 patk.py --dataset ${dataset} --output_dir ${dataset}_output/ --architecture ${architecture}
