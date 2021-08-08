import json

doc2text = {}
with open('../MAG/test_text.txt') as fin:
	for idx, line in enumerate(fin):
		data = line.strip().split('\t')
		doc = data[0]
		text = data[1]
		doc2text[doc] = text

label2text = {}
with open('../MAG/comb_text.txt') as fin:
	for idx, line in enumerate(fin):
		data = line.strip().split('\t')
		label = data[0]
		text = data[1]
		label2text[label] = text

with open('../LabelMatching_MAG/mag_filter.json') as fin, open('mag_test/test_original.txt', 'w') as fout:
	for idx, line in enumerate(fin):
		js = json.loads(line)
		doc_text = doc2text[js['paper']]

		labels = js['matched_id']
		
		for label in labels:
			label_text = label2text[label]
			fout.write('1\t'+doc_text+'\t'+label_text+'\n')
