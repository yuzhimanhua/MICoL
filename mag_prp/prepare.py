import json
from collections import defaultdict
import random

dataset = 'MAG'
metadata = 'reference'

doc2text = {}
docs = []
with open(f'../{dataset}/train_text.txt') as fin:
	for idx, line in enumerate(fin):
		data = line.strip().split('\t')
		doc = data[0]
		text = data[1]
		doc2text[doc] = text
		docs.append(doc)

meta2doc = defaultdict(set)
doc2meta = {}
with open(f'../{dataset}/{dataset}_train.json') as fin:
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
with open('dataset.txt', 'w') as fout:
	for idx, doc in enumerate(doc2meta):
		# sample positive
		metas = doc2meta[doc]
		dps = []
		for meta in metas:
			candidates = list(meta2doc[meta])
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
