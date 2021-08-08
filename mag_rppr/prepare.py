import json
from collections import defaultdict
import random

metadata1 = 'citation'
metadata2 = 'citation'

doc2text = {}
docs = []
with open('../../MAG/train_text.txt') as fin:
	for idx, line in enumerate(fin):
		data = line.strip().split('\t')
		doc = data[0]
		text = data[1]
		doc2text[doc] = text
		docs.append(doc)

meta12doc = defaultdict(set)
doc2meta1 = {}
doc2meta2 = {}
with open('../../MAG/MAG_train.json') as fin:
	for idx, line in enumerate(fin):
		js = json.loads(line)
		doc = js['paper']

		meta1s = js[metadata1]
		if not isinstance(meta1s, list):
			meta1s = [meta1s]
		for meta1 in meta1s:
			meta12doc[meta1].add(doc)
		doc2meta1[doc] = set(meta1s)

		meta2s = js[metadata2]
		if not isinstance(meta2s, list):
			meta2s = [meta2s]
		doc2meta2[doc] = set(meta2s)

tot = len(doc2meta1)
train_ratio = 0.9
fout = open('train.txt', 'w')
for idx, doc in enumerate(doc2meta1):
	if idx == int(tot*train_ratio):
		fout.close()
		fout = open('dev.txt', 'w')
	
	# sample positive
	meta1s = doc2meta1[doc]
	dps = []
	for meta1 in meta1s:
		candidates = []
		for d_cand in list(meta12doc[meta1]):
			if d_cand == doc:
				continue
			meta_intersec = doc2meta2[doc].intersection(doc2meta2[d_cand])
			if metadata1 != metadata2:
				if len(meta_intersec) >= 1:
					candidates.append(d_cand)
			else:
				if len(meta_intersec) >= 2:
					candidates.append(d_cand)
		if len(candidates) > 1:
			while True:
				dp = random.choice(candidates)
				if dp != doc:
					dps.append(dp)
					break
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