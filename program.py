import csv

import scipy
from entropymodel import EntropyModel

# Pothos 2000 Data
training = ['XXVT', 'XXVXJJ', 'VXJJ', 'XVJTVJ', 'XXVXJ', 'XVX', 'VXJJJJ', 'XVT', 'XXXVT', 'VJ', 'XVXJJJ', 'VJTVTV',
            'XXVJ', 'VTVJ', 'VJTVX', 'XXXVTV', 'XXVJ', 'VJTVX', 'XXXVTV', 'XVXJJ', 'VT', 'VJTVXJ', 'XXXVX', 'VJTXVJ',
            'XVXJ',
            'XXXXVX']
test = 'XXVXJ XVTV VXJ XXVTV XVJTVX XXVTVJ VJTXVX VX VJTVT VTVJJ VTVJ XVJTVT VTV XVTVJ XVTVJ VTVJJ VJTV XXV XVXV XVXVJ XXVJJJ XJJ VXVJ XVXT XXJJ VXJTJ XXWJJ JXVT XXTX TVJ VXJJX VJJXVT'.split(" ")

symbols = ['b', 'X', 'J', 'V', 'T', 'e']

emodel = EntropyModel()
bigram_list = emodel.training_to_bigrams(training)
pp_data_filename = "pothos2000.csv"

# construct entropy dictionary
entropy_dict = {}
bigram_prob = {}
for s in symbols[:-1]:
    freq_list = emodel.bigram_frequency(s, bigram_list, symbols)
    prob_list = emodel.bigram_prob(freq_list)
    bigram_prob.update(prob_list)
    entropy = emodel.bigram_entropy(prob_list)

    entropy_dict[s] = entropy

print(bigram_prob)
print(entropy_dict)

# load data and test items
data = []
test = []
with open(pp_data_filename, "r") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        t = row['items']
        test.append(t)
        t = t.upper()
        data.append((t, float(row['p1']), emodel.average_entropy(t, entropy_dict, bigram_prob),
                     emodel.sum_entropy(t, entropy_dict, bigram_prob),
                     emodel.avg_prob(t, bigram_prob)))

pp_data = [item[1] for item in data]
avg_entropy_predict = [item[2] for item in data]
sum_entropy_predict = [item[3] for item in data]
prob_predict = [item[4] for item in data]

cor_avgE = scipy.stats.pearsonr(pp_data, avg_entropy_predict)
cor_sumE = scipy.stats.pearsonr(pp_data, sum_entropy_predict)
cor_prob = scipy.stats.pearsonr(pp_data, prob_predict)

print(data)
print(cor_avgE)
print(cor_sumE)
print(cor_prob)




