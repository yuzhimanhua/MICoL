import json
import math
import os
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--output_dir', required=True, type=str)
parser.add_argument('--architecture', required=True, type=str)
args = parser.parse_args()

def get_inv_propensity(y, tot, a=0.55, b=1.5):
	c = (math.log(tot) - 1) * ((b + 1) ** a)
	return 1.0 + c * (label2cnt[y] + b) ** (-a)

label2cnt = defaultdict(int)
tot_train = 0
with open('MAG/MAG_train.json') as fin:
	for idx, line in enumerate(fin):
		js = json.loads(line)
		for y in js['label']:
			label2cnt[y] += 1
		tot_train += 1

p1 = p3 = p5 = 0
n3 = n5 = 0
psp1 = psp3 = psp5 = 0
psn3 = psn5 = 0
tot = 0
with open(os.path.join(args.output_dir, f'prediction_{args.architecture}.json')) as fin:
	for idx, line in enumerate(fin):
		prec1 = prec3 = prec5 = 0
		psprec1 = psprec3 = psprec5 = 0
		dcg3 = dcg5 = 0
		psdcg3 = psdcg5 = 0
		idcg3 = idcg5 = 0

		js = json.loads(line)
		y = js['label']
		y_pred = [x[0] for x in js['predicted_label']]
		for rank, label in enumerate(y_pred[:5]):
			if label in y:
				invp = get_inv_propensity(label, tot_train)
				if rank < 1:
					prec1 += 1
					psprec1 += invp
				if rank < 3:
					prec3 += 1
					psprec3 += invp
					dcg3 += 1/math.log2(rank+2)
					psdcg3 += invp/math.log2(rank+2)
				if rank < 5:
					prec5 += 1
					psprec5 += invp
					dcg5 += 1/math.log2(rank+2)
					psdcg5 += invp/math.log2(rank+2)

		for rank in range(min(3, len(y))):
			idcg3 += 1/math.log2(rank+2)
		for rank in range(min(5, len(y))):
			idcg5 += 1/math.log2(rank+2)

		p1 += prec1
		p3 += prec3/3
		p5 += prec5/5
		n3 += dcg3/idcg3
		n5 += dcg5/idcg5

		psp1 += psprec1
		psp3 += psprec3/3
		psp5 += psprec5/5
		psn3 += psdcg3/idcg3
		psn5 += psdcg5/idcg5

		tot += 1

print(p1/tot, p3/tot, p5/tot, n3/tot, n5/tot)
print(psp1/tot, psp3/tot, psp5/tot, psn3/tot, psn5/tot)
