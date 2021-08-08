import json

paths = ['pap', 'pr', 'rp', 'prp', 'rpr', 'paap', 'pavp', 'prrp', 'rppr', 'eda']

# model = 'bi'
# thrss = [0.1*x for x in range(-10, 11)]

model = 'cross'
thrss = [0.5*x for x in range(-40, 41)]

for path in paths:
	print(path)
	best_f1 = 0.0
	best_thrs = None
	for thrs in thrss:
		tot = 0
		tot_f1 = 0.0
		with open(f'mag_output_{path}/prediction_{model}.json') as fin:
			for line in fin:
				js = json.loads(line)
				lg = set(js['label'])
				lp = set()
				for tup in js['matched_id']:
					if tup[1] < thrs:
						break
					lp.add(tup[0])

				c = lg.intersection(lp)
				tot_f1 += 2*len(c) / (len(lg)+len(lp))
				tot += 1

		if tot_f1/tot > best_f1:
			best_f1 = tot_f1/tot
			best_thrs = thrs

	print(best_f1, best_thrs)