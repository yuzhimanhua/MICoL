import json
import argparse
import os

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--output_dir', required=True, type=str)
parser.add_argument('--architecture', required=True, type=str)
args = parser.parse_args()

pred = []
with open(os.path.join(args.output_dir, f'prediction_{args.architecture}.txt')) as fin:
	for line in fin:
		data = float(line.strip())
		pred.append(data)

i = 0
with open('../BM25/mag_filter.json') as fin, open(os.path.join(args.output_dir, f'prediction_{args.architecture}.json'), 'w') as fout:
	for line in fin:
		js = json.loads(line)
		out = {}
		out['paper'] = js['paper']
		out['label'] = js['label']

		labels = js['predicted_label']
		l = len(labels)
		sim = {}
		for label, score in zip(labels, pred[i:i+l]):
			sim[label] = score
		sim_sorted = sorted(sim.items(), key=lambda x:x[1], reverse=True)
		out['predicted_label'] = sim_sorted
		fout.write(json.dumps(out)+'\n')

		i += l
