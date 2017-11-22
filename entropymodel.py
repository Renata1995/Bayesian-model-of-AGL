import math
import scipy
import csv


class EntropyModel:

    def training_to_bigrams(self, training):
        """
        Transform a list of training instances into a list of bigrams.
        Each training instance X is changed to bXe.
        'b' denotes the start symbol.
        'e' denotes the end symbol
        E.g.
        training: ['ABB','BBC', 'BCA']
        bigram_list: ['bA', 'AB', 'BB', 'Be', 'bB', BB', 'BC', 'Ce', 'bB', 'BC', 'CA', 'Ae']

        :param training: a list of training instance strings
        :return: a list of bigrams
        """

        # transform into bigrams
        bigram_list = []
        for item in training:
            bigram_list.extend(self.string_to_bigram(item))

        return bigram_list

    def string_to_bigram (self, str):
        """
        Change a string into a list of bigrams.
        The string str is changed to bstre.
        'b' denotes the start symbol.
        'e' denotes the end symbol

        E.g.
        str: 'ABB'
        bigrams: ['bA', 'AB', 'BB', 'Be']
        """
        str = 'b' + str + 'e'

        bigrams = []
        for i in range(0, len(str)-1):
            bg = str[i: i+2]
            bigrams.append(bg)

        return bigrams

    def bigram_frequency(self, start_symbol, bigram_list, symbols):
        """
        Calculate the frequency of each bigram starts with a given start symbol.
        E.g.
        start_symbol = 'A'
        symbols = ['A', 'B', 'C', 'e']
        bigram_list: ['bA', 'AB', 'BB', 'Be', 'bB', BB', 'BC', 'Ce', 'bB', 'BC', 'CA', 'Ae']
        freq_list = {'AA': 0, 'AB':1, 'AC': 0, 'Ae': 1}
        """

        # init the frequency dictionary
        freq_list = {}
        for s in symbols[1:]:
            key = start_symbol + s
            freq_list[key] = 0

        # add frequency
        for item in bigram_list:
            if item.startswith(start_symbol):
                freq_list[item] = freq_list[item] + 1

        return freq_list

    def bigram_prob(self, freq_dict):
        """
        Calculate the probability of each unique bigram start with a given symbol
        :param freq_dict: frequency of each unique bigram start with a given symbol
        E.g.
        freq_list = {'AA': 0, 'AB':1, 'AC': 0, 'Ae': 1}
        prob_list = {'AA': 0, 'AB':0.5, 'AC': 0, 'Ae': 0.5}
        """
        # calculate the sum of all bigrams start with a given symbol
        freq_sum = sum([freq_dict[key] for key in freq_dict.keys()])

        # convert frequency to probability
        prob_list = {}
        for key in freq_dict.keys():
            prob_list[key] = freq_dict[key] / freq_sum

        return prob_list


    def bigram_entropy(self, prob_dict):
        """
        Calculate the entropy of bigrams start with a given symbol
        :param prob_dict: probability of each unique bigram start with a given symbol
        :return: an entropy value
        E.g.
        prob_list = {'AA': 0, 'AB':0.5, 'AC': 0, 'Ae': 0.5}
        entropy = - (0*2 + 0.5log2(0.5)*2)
        """
        entropy = 0
        for bigram in prob_dict.keys():
            prob = prob_dict[bigram]
            if (prob == 0):
                continue
            else:
                entropy += (prob * (math.log2(prob)))

        return -1 * entropy


    def sum_entropy(self, test_item, edict, bigram_prob):
        """
        Sum of bigram entropy for a test item
        :param test_item:  test item
        :param edict:  bigram entropy list
        :return: an entropy value

        E.g.
        test_item = 'ABB'
        edict = {'bA':1.2, 'AB': 1.3, 'BB': 0.9, 'Be': 0.4,....}
        entropy = E('bA') + E('AB') + E('BB') + E('Be') = 1.2 + 1.3 + 0.9 + 0.4 = 3.8
        """
        sum_entropy = 0
        test_bigram = self.string_to_bigram(test_item)

        for bigram in test_bigram:
            if bigram_prob[bigram] == 0:
                prob = 1/len(edict)
                sum_entropy += (-len(edict) * prob * math.log2(prob))
            else:
                start_symbol = bigram[0]
                sum_entropy += edict[start_symbol]

        return sum_entropy

    def average_entropy(self, test_item, edict, bigram_prob):
        """
        Calculate the average bigram entropy of a test item
        """
        return self.sum_entropy(test_item, edict, bigram_prob) / (len(test_item) + 1)

    def avg_prob(self, test_item, prob_dict):
        """

        :param test_item:
        :param prob_dict:
        :return:
        """
        bigram_list = self.string_to_bigram(test_item)

        sum_prob = 0
        for bigram in bigram_list:
            sum_prob += prob_dict[bigram]

        return sum_prob/(len(bigram_list))