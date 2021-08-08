import json
from collections import defaultdict
import random

metadata = 'reference'

doc2text = {}
docs = []
with open('../../MAG/train_text.txt') as fin:
	for idx, line in enumerate(fin):
		data = line.strip().split('\t')
		doc = data[0]
		text = data[1]
		doc2text[doc] = text
		docs.append(doc)

meta2doc = defaultdict(set)
doc2meta = {}
with open('../../MAG/MAG_train.json') as fin:
	for idx, line in enumerate(fin):
		js = json.loads(line)
		doc = js['paper']
		metas = js[metadata]
		if not isinstance(metas, list):
			metas = [metas]
		for meta in metas:
			meta2doc[meta].add(doc)
		doc2meta[doc] = set(metas)

tot = len(doc2meta)
train_ratio = 0.9
fout = open('train.txt', 'w')
for idx, doc in enumerate(doc2meta):
	if idx == int(tot*train_ratio):
		fout.close()
		fout = open('dev.txt', 'w')
	
	# sample positive
	dps = [x for x in doc2meta[doc] if x in doc2text]
	if len(dps) == 0:
		continue
	dp = random.choice(dps)

	# sample negative (for validation only)
	while True:
		dn = random.choice(docs)
		if dn != doc and dn != dp:
			break	
			
	fout.write('1\t'+doc2text[doc]+'\t'+doc2text[dp]+'\n')
	fout.write('0\t'+doc2text[doc]+'\t'+doc2text[dn]+'\n')

fout.close()