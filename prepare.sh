dataset=MAG
metagraph=PRP

echo "=====Step 1: Preparing Testing Data====="
python prepare_test.py --dataset ${dataset}

echo "=====Step 2: Generating Training Data====="
python prepare_train.py --dataset ${dataset} --metagraph ${metagraph}

head -100000 ${dataset}_input/dataset.txt > ${dataset}_input/train.txt
sed -n '100001,110000p' ${dataset}_input/dataset.txt > ${dataset}_input/dev.txt