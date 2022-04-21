import json
import argparse
import os

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='MAG', type=str)
args = parser.parse_args()

dataset = args.dataset
if not os.path.exists(f'{dataset}_input/'):
	os.mkdir(f'{dataset}_input/')

doc2text = {}
with open(f'{dataset}/{dataset}_test.json') as fin:
	for line in fin:
		data = json.loads(line)
		doc = data['paper']
		text = data['text'].replace('_', ' ')
		doc2text[doc] = text

label2text = {}
with open(f'{dataset}/{dataset}_label.json') as fin:
	for line in fin:
		data = json.loads(line)
		label = data['label']
		text = data['combined_text']
		label2text[label] = text

with open(f'{dataset}/{dataset}_candidates.json') as fin, open(f'{dataset}_input/test.txt', 'w') as fout:
	for line in fin:
		data = json.loads(line)
		doc_text = doc2text[data['paper']]	
		labels = data['predicted_label']
		for label in labels:
			label_text = label2text[label]
			fout.write(f'1\t{doc_text}\t{label_text}\n')
