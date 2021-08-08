import json
import math
import os
import argparse

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--output_dir', required=True, type=str)
parser.add_argument('--architecture', required=True, type=str)
args = parser.parse_args()

p1 = p3 = p5 = 0
n3 = n5 = 0
tot = 0
with open(os.path.join(args.output_dir, f'prediction_{args.architecture}.json')) as fin:
	for idx, line in enumerate(fin):
		prec1 = prec3 = prec5 = 0
		dcg3 = dcg5 = 0
		idcg3 = idcg5 = 0

		js = json.loads(line)
		y = js['label']
		y_pred = [x[0] for x in js['matched_id']]
		for rank, label in enumerate(y_pred[:5]):
			if label in y:
				if rank < 1:
					prec1 += 1
				if rank < 3:
					prec3 += 1
					dcg3 += 1/math.log2(rank+2)
				if rank < 5:
					prec5 += 1
					dcg5 += 1/math.log2(rank+2)

		for rank in range(min(3, len(y))):
			idcg3 += 1/math.log2(rank+2)
		for rank in range(min(5, len(y))):
			idcg5 += 1/math.log2(rank+2)

		p1 += prec1
		p3 += prec3/3
		p5 += prec5/5
		n3 += dcg3/idcg3
		n5 += dcg5/idcg5
		tot += 1

print(p1/tot, p3/tot, p5/tot, n3/tot, n5/tot)
