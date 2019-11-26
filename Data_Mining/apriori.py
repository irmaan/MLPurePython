


import numpy as np 
import pandas as pd



class Apriori:
    
    def __init__(self, transactions, min_support, min_confidence):
        self.transactions = transactions
        self.min_support = min_support # The minimum support.
        self.min_confidence = min_confidence # The minimum confidence.
        self.support_data = {} # A dictionary. The key is frequent itemset and the value is support.


        
  ## create frequent candidate 1-itemset C1 by scaning data set.
    def create_C1(self):
        C1 = set()
        for transaction in self.transactions:
            for item in transaction:
                C1.add(frozenset([item]))
        return C1


    
  ## Create Ck.
    def create_Ck(self, Lksub1, k):

      ## Lksub1: Lk-1, a set which contains all frequent candidate (k-1)-itemsets. k: the item number of a frequent itemset.  
      ## Ck: A set which contains all all frequent candidate k-itemsets.

        Ck = set()
        len_Lksub1 = len(Lksub1)
        list_Lksub1 = list(Lksub1)
        for i in range(len_Lksub1):
            for j in range(1, len_Lksub1):
                l1 = list(list_Lksub1[i])
                l2 = list(list_Lksub1[j])
                l1.sort()
                l2.sort()
                if l1[0:k-2] == l2[0:k-2]:
                    # TODO: self joining Lk-1
                    Ck_item = list_Lksub1[i] | list_Lksub1[j]
                    # TODO: pruning
                    flag = 1
                    for item in Ck_item:
                        sub_Ck = Ck_item - frozenset([item])
                        if sub_Ck not in Lksub1:
                            flag = 0
                    if flag == 1:
                        Ck.add(Ck_item)

        return Ck

    
    ##Generate Lk by executing a delete policy from Ck.

    def generate_Lk_from_Ck(self, Ck):

        ## Ck: A set which contains all all frequent candidate k-itemsets.
        ## Lk: A set which contains all all frequent k-itemsets.
               
        Lk = set()
        item_count = {}
        for transaction in self.transactions:
            for item in Ck:
                if item.issubset(transaction):
                    if item not in item_count:
                        item_count[item] = 1
                    else:
                        item_count[item] += 1
        t_num = float(len(self.transactions))
        for item in item_count:
            support = item_count[item] / t_num
            if support >= self.min_support:
                Lk.add(item)
                self.support_data[item] = support
        return Lk


    ##Generate all frequent item sets..

    def generate_L(self):
          
        self.support_data = {}
        
        C1 = self.create_C1()
        L1 = self.generate_Lk_from_Ck(C1)
        Lksub1 = L1.copy()
        L = []
        L.append(Lksub1)
        i = 2
        while True:
            Ci = self.create_Ck(Lksub1, i)
            Li = self.generate_Lk_from_Ck(Ci)
            if Li:
                Lksub1 = Li.copy()
                L.append(Lksub1)
                i += 1
            else:
                break
        return L
        
        
    ## Generate association rules from frequent itemsets.
    def generate_rules(self):
        
      ## big_rule_list: A list which contains all big rules. Each big rule is represented as a 3-tuple.
      
        L = self.generate_L()
        big_rule_list = []
        sub_set_list = []
        for i in range(0, len(L)):
            for freq_set in L[i]:
                for sub_set in sub_set_list:
                    if sub_set.issubset(freq_set):
                        # TODO : compute the confidence
                        conf = self.support_data[freq_set] / self.support_data[freq_set - sub_set]
                        big_rule = (freq_set - sub_set, sub_set, conf)
                        if conf >= self.min_confidence and big_rule not in big_rule_list:
                            big_rule_list.append(big_rule)
                sub_set_list.append(freq_set)
        return big_rule_list

# Data Preparation

data = pd.read_csv('/content/sample_data/GroceryStoreDataSet.csv', header=None)
data.head()
transactions = []
for i in range(len(data)):
     transactions.append(data.values[i, 0].split(','))
print(transactions)

 ## Test Algorithm

# 1. Model construction
# """

model = Apriori(transactions, min_support=0.1, min_confidence=0.75)

#  2. Frequent item set mining

# The algorithm generates a list of candidate itemsets, which includes all of the itemsets appearing within the dataset. Of the candidate itemsets generated, an itemset can be determined to be frequent if the number of transactions that it appears in is greater than the support value.
# """

L = model.generate_L()

for Lk in L:
     print('frequent {}-itemsets：\n'.format(len(list(Lk)[0])))

    for freq_set in Lk:
         print(freq_set, 'support:', model.support_data[freq_set])
    
     print()

# """3. Association rule mining

# Association rules can then trivially be generated by traversing the frequent itemsets, and computing associated confidence levels. Confidence is the proportion of the transactions containing item A which also contains item B.
# """

rule_list = model.generate_rules()

for item in rule_list:
    print(item[0], "=>", item[1], "confidence: ", item[2])
