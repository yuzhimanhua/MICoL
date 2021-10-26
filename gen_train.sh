dataset="mag"
path="prp"

cd ${dataset}_${path}/
python prepare.py

head -100000 dataset.txt > train.txt
sed -n '100001,110000p' dataset.txt > dev.txt