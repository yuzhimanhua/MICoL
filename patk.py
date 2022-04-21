import json
import numpy as np
from functools import partial
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from typing import Union, Optional, List, Iterable, Hashable
import os
import argparse

import warnings
warnings.filterwarnings('ignore')

TPredict = np.ndarray
TTarget = Union[Iterable[Iterable[Hashable]], csr_matrix]
TMlb = Optional[MultiLabelBinarizer]
TClass = Optional[List[Hashable]]

def get_mlb(classes: TClass = None, mlb: TMlb = None, targets: TTarget = None):
	if classes is not None:
		mlb = MultiLabelBinarizer(classes, sparse_output=True)
	if mlb is None and targets is not None:
		if isinstance(targets, csr_matrix):
			mlb = MultiLabelBinarizer(range(targets.shape[1]), sparse_output=True)
			mlb.fit(None)
		else:
			mlb = MultiLabelBinarizer(sparse_output=True)
			mlb.fit(targets)
	return mlb


def get_precision(prediction: TPredict, targets: TTarget, mlb: TMlb = None, classes: TClass = None, top=5):
	mlb = get_mlb(classes, mlb, targets)
	if not isinstance(targets, csr_matrix):
		targets = mlb.transform(targets)
	prediction = mlb.transform(prediction[:, :top])
	return prediction.multiply(targets).sum() / (top * targets.shape[0])

get_p_1 = partial(get_precision, top=1)
get_p_3 = partial(get_precision, top=3)
get_p_5 = partial(get_precision, top=5)


def get_ndcg(prediction: TPredict, targets: TTarget, mlb: TMlb = None, classes: TClass = None, top=5):
	mlb = get_mlb(classes, mlb, targets)
	log = 1.0 / np.log2(np.arange(top) + 2)
	dcg = np.zeros((targets.shape[0], 1))
	if not isinstance(targets, csr_matrix):
		targets = mlb.transform(targets)
	for i in range(top):
		p = mlb.transform(prediction[:, i: i+1])
		dcg += p.multiply(targets).sum(axis=-1) * log[i]
	return np.average(dcg / log.cumsum()[np.minimum(targets.sum(axis=-1), top) - 1])

get_n_3 = partial(get_ndcg, top=3)
get_n_5 = partial(get_ndcg, top=5)


def get_inv_propensity(train_y: csr_matrix, a=0.55, b=1.5):
	n, number = train_y.shape[0], np.asarray(train_y.sum(axis=0)).squeeze()
	c = (np.log(n) - 1) * ((b + 1) ** a)
	return 1.0 + c * (number + b) ** (-a)


def get_psp(prediction: TPredict, targets: TTarget, inv_w: np.ndarray, mlb: TMlb = None,
			classes: TClass = None, top=5):
	mlb = get_mlb(classes, mlb)
	if not isinstance(targets, csr_matrix):
		targets = mlb.transform(targets)
	prediction = mlb.transform(prediction[:, :top]).multiply(inv_w)
	num = prediction.multiply(targets).sum()
	t, den = csr_matrix(targets.multiply(inv_w)), 0
	for i in range(t.shape[0]):
		den += np.sum(np.sort(t.getrow(i).data)[-top:])
	return num / den

get_psp_1 = partial(get_psp, top=1)
get_psp_3 = partial(get_psp, top=3)
get_psp_5 = partial(get_psp, top=5)


def get_psndcg(prediction: TPredict, targets: TTarget, inv_w: np.ndarray, mlb: TMlb = None,
			   classes: TClass = None, top=5):
	mlb = get_mlb(classes, mlb)
	log = 1.0 / np.log2(np.arange(top) + 2)
	psdcg = 0.0
	if not isinstance(targets, csr_matrix):
		targets = mlb.transform(targets)
	for i in range(top):
		p = mlb.transform(prediction[:, i: i+1]).multiply(inv_w)
		psdcg += p.multiply(targets).sum() * log[i]
	t, den = csr_matrix(targets.multiply(inv_w)), 0.0
	for i in range(t.shape[0]):
		num = min(top, len(t.getrow(i).data))
		den += -np.sum(np.sort(-t.getrow(i).data)[:num] * log[:num])
	return psdcg / den

get_psndcg_3 = partial(get_psndcg, top=3)
get_psndcg_5 = partial(get_psndcg, top=5)



parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', required=True, type=str)
parser.add_argument('--output_dir', required=True, type=str)
parser.add_argument('--architecture', required=True, type=str)
args = parser.parse_args()

preds = []
targets = []
with open(os.path.join(args.output_dir, f'prediction_{args.architecture}.json')) as fin:
	for line in fin:
		data = json.loads(line)
		targets.append(data['label'])
		pred = [x[0] for x in data['predicted_label']]
		pred = pred[:5] + ['PAD']*(5-len(pred))
		preds.append(pred)
preds = np.array(preds)

mlb = MultiLabelBinarizer(sparse_output=True)
targets = mlb.fit_transform(targets)
print('P@1:', get_p_1(preds, targets, mlb), ', ', \
	  'P@3:', get_p_3(preds, targets, mlb), ', ', \
	  'P@5:', get_p_5(preds, targets, mlb), ', ', \
	  'NDCG@3:', get_n_3(preds, targets, mlb), ', ', \
	  'NDCG@5:', get_n_5(preds, targets, mlb))


train_labels = []
with open(f'{args.dataset}/{args.dataset}_train.json') as fin:
	for line in fin:
		data = json.loads(line)
		train_labels.append(data['label'])
inv_w = get_inv_propensity(mlb.transform(train_labels), 0.55, 1.5)
print('PSP@1:', get_psp_1(preds, targets, inv_w, mlb), ', ', \
	  'PSP@3:', get_psp_3(preds, targets, inv_w, mlb), ', ', \
	  'PSP@5:', get_psp_5(preds, targets, inv_w, mlb), ', ', \
	  'PSN@3:', get_psndcg_3(preds, targets, inv_w, mlb), ', ', \
	  'PSN@5:', get_psndcg_5(preds, targets, inv_w, mlb))
