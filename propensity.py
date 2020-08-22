from sklearn import linear_model
from scipy.sparse import coo_matrix
from scipy.stats import ttest_rel, binom
#from scipy.stats import chisqprob # old version
from scipy.stats import chi2
import numpy as np
from math import log
import sys

filename = sys.argv[1]
param_reg = float(sys.argv[2])
param_thresh = float(sys.argv[3])

# Create inverted index of document indices for each word from input file

word_docs = {}
vocab = {}
l = 0
for line in open(filename):
	tokens = line.strip().split() 
	for word in set(tokens[1:]):
		if word not in vocab:
			vocab[word] = len(vocab)

		if word not in word_docs:
			word_docs[word] = []
		word_docs[word].append(l)

	l += 1

L = l
V = len(vocab)
print('%d documents, %d distinct word types' % (L, V))

# Run matching routine for each word to calculate its p-value
 
model = linear_model.LogisticRegression(C=param_reg)

probscores = {}
tscores = {}
chiscores = {}

for treatment in sorted(vocab):
	rows = []
	cols = []
	values = []
	y = []

	# Read through input file and create a sparse matrix of word counts (excluding treatment word)
	# to train the propensity classifier

	l = 0
	for line in open(filename):
		tokens = line.strip().split() 
	
		contains_treatment = 0
		for word in set(tokens[1:]):
			v = vocab[word]
			
			if word == treatment:
				contains_treatment = 1
			else:
				rows.append(l)
				cols.append(v)
				values.append(1.)
		
		y.append(contains_treatment)
		l += 1
	
	data = coo_matrix((values, (rows, cols)), shape=(l, V)).toarray()
	
	model.fit(data, y)
	scores = model.predict_proba(data)

	# Read through the file again and create a list of propensity scores (from the classifier above)
	# and document labels (the first token of each document)

	l = 0
	docs = []
	for line in open(filename):
		tokens = line.strip().split() 
	
		label = int(tokens[0])
		contains_treatment = y[l]

		docs.append((scores[l][1], label, contains_treatment))
			
		l += 1

	# This code corresponds to the algorithm described in the "implementation" paragraph 
	# of section 3.1 in the paper
	
	D = len(docs)
	prop_scores = []
	buckets = {}
	for i in range(D):
		score = docs[i][0]
		contains_treatment = docs[i][2]
		prop_scores.append(score)

		bucket = round(score, 2)
		if bucket not in buckets:
			buckets[bucket] = {}
			buckets[bucket][0] = []
			buckets[bucket][1] = []
		buckets[bucket][contains_treatment].append(i)

	propSD = np.std(prop_scores)
	threshold = param_thresh * propSD
	rounded = round(threshold, 2)

	if rounded > threshold:
		threshold = rounded
	else:
		threshold = rounded + 0.01

	# Now do the matching

	labels_treatment = []
	labels_control = []
	matched = set()

	il = 0
	for i in word_docs[treatment]:
		score = docs[i][0]
		label = docs[i][1]

		il += 1

		best_match = -1
		best_diff = 100.

		# TODO: create the bucket list earlier so it's not recreated every time
		# create pointers for the list index for each possible bucket, 
		# and an end point based on the threshold, and then slice
		bucket = round(score, 2)
		bucket_list = [bucket]
		b = bucket
		bn = bucket
		while abs(b - bucket) < threshold:
			b = round(b + 0.01, 2)
			bn = round(bn - 0.01, 2)

			if not b > 1.:
				bucket_list.append(b)
			if not bn < 0.:
				bucket_list.append(bn)

			if b > 1. and bn < 0.:
				break


		d = -1
		for bucket in bucket_list:
			if bucket in buckets and len(buckets[bucket][0]) > 0:
				d = buckets[bucket][0].pop()
				break

			
		if d == -1: continue # no match

		score2 = docs[d][0]
		label2 = docs[d][1]

		labels_treatment.append(label)
		labels_control.append(label2)

	# Now calculate the sufficient statistics from the matched samples
	# and use them to calculate p-values 

	if len(labels_treatment) == 0:
		chiscores[treatment] = 1.0 
	else:
		tscore = ttest_rel(labels_treatment, labels_control)[0]

		if str(tscore) == 'nan': 
			tscore = 0.

		tscores[treatment] = tscore 
		probscores[treatment] = np.mean(labels_treatment) - np.mean(labels_control)

		treatment_neg = 0
		for label in labels_treatment:
			if label == 0:
				treatment_neg += 1
		notreatment_pos = 0
		for label in labels_control:
			if label == 1:
				notreatment_pos += 1

		if treatment_neg + notreatment_pos == 0:
			pvalue = 1.0 
		else:
			treatment_neg = float(treatment_neg)
			notreatment_pos = float(notreatment_pos)

			n = treatment_neg + notreatment_pos

			# use binomial approximation for small n
			if n < 0:
				pvalue = 2.0 * binom.cdf(min(treatment_neg, notreatment_pos), treatment_neg + notreatment_pos, 0.5)
				if pvalue > 1.: pvalue = 1.
			else:
				chi_score = pow(treatment_neg - notreatment_pos, 2) / (treatment_neg + notreatment_pos)
				#pvalue = chisqprob(chi_score, 1) # old version
				pvalue = chi2.sf(chi_score, 1)

		if pvalue == 0.:
			pvalue = float('-inf')
		else:
			pvalue = log(pvalue)

		chiscores[treatment] = pvalue 

# Write log-p-values to output file: same filename as input, with ".out" appended to end

outfile = open('%s.out' % filename, 'w')
for word in sorted(chiscores, key=chiscores.get, reverse=False):
	outfile.write('%s %.20f\n' % (word, chiscores[word]))


