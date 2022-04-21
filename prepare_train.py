import json
import argparse
import os
from collections import defaultdict
import random

# P->P and P<-P
def no_intermediate_node(dataset, doc2text, docs, metadata):
	meta2doc = defaultdict(set)
	doc2meta = {}
	with open(f'{dataset}/{dataset}_train.json') as fin:
		for idx, line in enumerate(fin):
			if idx % 100000 == 0:
				print(idx)
			data = json.loads(line)
			doc = data['paper']

			metas = data[metadata]
			if not isinstance(metas, list):
				metas = [metas]
			for meta in metas:
				meta2doc[meta].add(doc)
			doc2meta[doc] = set(metas)

	with open(f'{dataset}_input/dataset.txt', 'w') as fout:
		for idx, doc in enumerate(doc2meta):
			if idx % 100000 == 0:
				print(idx)

			# sample positive
			dps = [x for x in doc2meta[doc] if x in doc2text]
			if len(dps) == 0:
				continue
			dp = random.choice(dps)

			# sample negative
			while True:
				dn = random.choice(docs)
				if dn != doc and dn != dp:
					break	
					
			fout.write(f'1\t{doc2text[doc]}\t{doc2text[dp]}\n')
			fout.write(f'0\t{doc2text[doc]}\t{doc2text[dn]}\n')


# PAP, PVP, P->P<-P, and P<-P->P
def one_intermediate_node(dataset, doc2text, docs, metadata):
	meta2doc = defaultdict(set)
	doc2meta = {}
	with open(f'{dataset}/{dataset}_train.json') as fin:
		for idx, line in enumerate(fin):
			if idx % 100000 == 0:
				print(idx)
			data = json.loads(line)
			doc = data['paper']

			metas = data[metadata]
			if not isinstance(metas, list):
				metas = [metas]
			for meta in metas:
				meta2doc[meta].add(doc)
			doc2meta[doc] = set(metas)

	with open(f'{dataset}_input/dataset.txt', 'w') as fout:
		for idx, doc in enumerate(doc2meta):
			if idx % 100000 == 0:
				print(idx)

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

			# sample negative
			while True:
				dn = random.choice(docs)
				if dn != doc and dn != dp:
					break	
					
			fout.write(f'1\t{doc2text[doc]}\t{doc2text[dp]}\n')
			fout.write(f'0\t{doc2text[doc]}\t{doc2text[dn]}\n')


# P(AA)P, P(AV)P, P->(PP)<-P, and P<-(PP)->P
def two_intermediate_node(dataset, doc2text, docs, metadata1, metadata2):
	meta12doc = defaultdict(set)
	doc2meta1 = {}
	doc2meta2 = {}
	with open(f'{dataset}/{dataset}_train.json') as fin:
		for idx, line in enumerate(fin):
			if idx % 100000 == 0:
				print(idx)
			data = json.loads(line)
			doc = data['paper']

			meta1s = data[metadata1]
			if not isinstance(meta1s, list):
				meta1s = [meta1s]
			for meta1 in meta1s:
				meta12doc[meta1].add(doc)
			doc2meta1[doc] = set(meta1s)

			meta2s = data[metadata2]
			if not isinstance(meta2s, list):
				meta2s = [meta2s]
			doc2meta2[doc] = set(meta2s)

	with open(f'{dataset}_input/dataset.txt', 'w') as fout:
		for idx, doc in enumerate(doc2meta1):
			if idx % 100000 == 0:
				print(idx)

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

			# sample negative
			while True:
				dn = random.choice(docs)
				if dn != doc and dn != dp:
					break	
					
			fout.write(f'1\t{doc2text[doc]}\t{doc2text[dp]}\n')
			fout.write(f'0\t{doc2text[doc]}\t{doc2text[dn]}\n')


parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='MAG', type=str)
parser.add_argument('--metagraph', default='PRP', type=str)
args = parser.parse_args()

dataset = args.dataset
metagraph = args.metagraph

doc2text = {}
docs = []
with open(f'{dataset}/{dataset}_train.json') as fin:
	for idx, line in enumerate(fin):
		if idx % 100000 == 0:
			print(idx)
		data = json.loads(line)
		doc = data['paper']
		text = data['text'].replace('_', ' ')
		doc2text[doc] = text
		docs.append(doc)

# P->P
if metagraph == 'PR':
	no_intermediate_node(dataset, doc2text, docs, 'reference')
# P<-P
elif metagraph == 'PC':
	no_intermediate_node(dataset, doc2text, docs, 'citation')
# PAP
elif metagraph == 'PAP':
	one_intermediate_node(dataset, doc2text, docs, 'author')
# PVP
elif metagraph == 'PVP':
	one_intermediate_node(dataset, doc2text, docs, 'venue')
# P->P<-P
elif metagraph == 'PRP':
	one_intermediate_node(dataset, doc2text, docs, 'reference')
# P<-P->P
elif metagraph == 'PCP':
	one_intermediate_node(dataset, doc2text, docs, 'citation')
# P(AA)P
elif metagraph == 'PAAP':
	two_intermediate_node(dataset, doc2text, docs, 'author', 'author')
# P(AV)P
elif metagraph == 'PAVP':
	two_intermediate_node(dataset, doc2text, docs, 'author', 'venue')
# P->(PP)<-P
elif metagraph == 'PRRP':
	two_intermediate_node(dataset, doc2text, docs, 'reference', 'reference')
# P<-(PP)->P
elif metagraph == 'PCCP':
	two_intermediate_node(dataset, doc2text, docs, 'citation', 'citation')
else:
	print('Wrong Meta-path/Meta-graph!!')