import numpy as np
import piecash
import functools
import pickle
import pprint

from treeinterpreter import treeinterpreter as ti

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
#from treeinterpreter import treeinterpreter as ti
from sklearn import metrics

PREDICTION_ACCOUNTS = [
    'Assets:DNB:Brukskonto',
    'Assets:DNB:Regningskonto',
    'Liabilities:Credit Cards:DNB Mastercard',
    'Liabilities:Credit Cards:Bank Norwegian',
]

def OFX_transactions(book):
    for tr in book.transactions:
        if tr.notes is None or not 'OFX ext. info: |' in tr.notes:
            continue
        if len(tr.splits) != 2:
            continue
        if not any(
            split.account.fullname in PREDICTION_ACCOUNTS
            for split in tr.splits
        ):
            continue
        yield tr

def Prediction_Transactions(book):
    for tr in book.transactions:
        if len(tr.splits) != 2:
            continue
        if not any(
            split.account.fullname in PREDICTION_ACCOUNTS
            for split in tr.splits
        ):
            continue
        yield tr

def split(delimiters, string, maxsplit=0):
    import re
    regexPattern = '|'.join(map(re.escape, delimiters))
    return re.split(regexPattern, string, maxsplit)


def extract_transaction_features(tr):
    delimiters = [" ", "/"]
    description = set(split(delimiters, tr.description))
    try:
        description.union(split(
            delimiters,
            tr.notes.split('OFX ext. info: |Memo:')[1]
        ))
    except AttributeError:
        pass
    except IndexError:
        if tr.notes != tr.description:
            description.union(split(
                delimiters,
                str(tr.notes
            )))
    return {
        'description': set(word for word in description if len(word) > 3),
        'account1': tr.splits[0].account.fullname,
        'account2': tr.splits[1].account.fullname,
        'amount': abs(functools.reduce(lambda x,y: x or y, [spl.quantity for spl in tr.splits])),
        'date': tr.post_date,
        'day': tr.post_date.day,
        'weekday': tr.post_date.isoweekday(),
}

def get_word_mapping(table):
    words = list(set(
        word.lower()
        for row in table
        for word in row['description']
    ))
    return words, {j:i for i, j in enumerate(words)}


DATA='data'
MAPPED_METADATA='mapped_metadata'
DATA='data'
MAPPED_METADATA='mapped_metadata'
MAPPED_DESCRIPTION='mapped_description'
ACCOUNT_LABELS='account_labels'
ACCOUNT_MAP='account_map'
WORDS='words'
WORD_MAP='word_map'
OUTPUT_ACCOUNTS='output_accounts'
MAPPED_OUTPUT_ACCOUNTS='mapped_output_accounts'

class TransactionFeatureSet(object):

    PREDICTION_DATA_LABELS = [
        DATA, MAPPED_METADATA, MAPPED_DESCRIPTION,
        ACCOUNT_LABELS, ACCOUNT_MAP, WORDS, WORD_MAP,
        OUTPUT_ACCOUNTS, MAPPED_OUTPUT_ACCOUNTS,
    ]

    def __init__(self, book_file, cache=True):
        self.book = piecash.open_book(args.file_, readonly=True)
        self.data = None
        if cache:
            try:
                with open('data_cache.json', 'rb') as file_:
                    pickled_data = pickle.load(file_)
                    self.data = {o:j for o,j in zip(self.PREDICTION_DATA_LABELS, pickled_data)}
            except IOError:
                pass
        if self.data is None:
            self.setup_data()
            with open("data_cache.json", 'wb') as file_:
                pickle.dump([self.data[label] for label in self.PREDICTION_DATA_LABELS], file_)

        self.fit_descriptions()
        description_features = self.get_description_probabilities(self.data[MAPPED_DESCRIPTION])
        self.full_features = np.hstack([
            self.data[MAPPED_METADATA],
            description_features,
        ])
        self.fit(self.full_features, self.data[MAPPED_OUTPUT_ACCOUNTS])

    def setup_data(self):
        data = [
            extract_transaction_features(tr)
            for tr in Prediction_Transactions(self.book)
        ]
        account_labels, acc_map, words, word_map = self.get_mappings(data)
        output_accounts, mapped_output_accounts = self.get_output_accounts(data, acc_map)
        self.data = {
            DATA: data,
            MAPPED_METADATA: self.get_mapped_features(data, acc_map),
            MAPPED_DESCRIPTION: self.get_mapped_description_features(data, word_map),
            ACCOUNT_LABELS: account_labels,
            ACCOUNT_MAP: acc_map,
            WORDS: words,
            WORD_MAP: word_map,
            OUTPUT_ACCOUNTS: output_accounts,
            MAPPED_OUTPUT_ACCOUNTS: mapped_output_accounts,
        }

    def fit(self, full_features, output_classes):
        #print(metadata_features[0])
        self.full_classifier = RandomForestClassifier()
        self.full_classifier.fit(full_features, output_classes)
        #print("Score full classifier: {}".format(self.full_classifier.score(full_features, self.data[MAPPED_OUTPUT_ACCOUNTS])))
        #print(self.full_classifier.predict_proba(full_features))
        #predicted = self.description_model.predict(self.data[MAPPED_DESCRIPTION])
        ##self.print_prediction(predicted)
        #instances = full_features[[0,4,10]]
        #prediction, bias, contributions = ti.predict(self.full_classifier, instances)
        #for i in range(len(instances)):
        #    print("Instance {}".format(i))
        #    print("Bias (trainset mean)".format(bias[i]))
        #    print("Feature contributions:")
        #    for c, feature in sorted(zip(contributions[i], 
        #                                 boston.feature_names), 
        #                             key=lambda x: -abs(x[0])):
        #        print("feature {}".format(round(c, 2)))
        #    print("-"*20)
        #prediction, bias, contributions = ti.predict(self.full_classifier, full_features[5])

    def print_prediction(self, predictions):
        predicted_labels = [self.data[ACCOUNT_LABELS][o] for o in predictions]
        output_labels = [self.data[ACCOUNT_LABELS][o] for o in self.data[MAPPED_OUTPUT_ACCOUNTS]]
        print('\n'.join(str(o) for o in zip(self.data[MAPPED_OUTPUT_ACCOUNTS], ['{} -> {}'.format(pred,out,) for pred, out in zip(predicted_labels, output_labels)])))

    def fit_descriptions(self):
        self.description_model = GaussianNB()
        self.description_model.fit(self.data[MAPPED_DESCRIPTION], self.data[MAPPED_OUTPUT_ACCOUNTS])
        #print("Score bayesian description only: {}".format(self.description_model.score(self.data[MAPPED_DESCRIPTION], self.data[MAPPED_OUTPUT_ACCOUNTS])))
        predicted = self.description_model.predict(self.data[MAPPED_DESCRIPTION])

    def get_description_probabilities(self, descriptions):
        return np.array(self.description_model.predict_proba(descriptions))
        
    def get_mappings(self, data):
        account_labels = [a.fullname for a in self.book.accounts]
        acc_map = { j:i for i,j in enumerate(account_labels)}
        words, word_map = get_word_mapping(data)
        return account_labels, acc_map, words, word_map

    def get_mapped_description_features(self, data, word_map):
        features = np.zeros((len(data), len(word_map)))
        for i, row in enumerate(data):
            for word, j in word_map.items():
                features[i][j] = 1 if word in row['description'] else 0
        return features

    def get_output_accounts(self, data, acc_map):
        output_accounts = [ row['account1'] for row in data ]
        mapped_output_accounts = np.zeros(len(data), dtype=np.int)
        for i, row in enumerate(data):
            mapped_output_accounts[i] = acc_map[row['account1']]
        return output_accounts, mapped_output_accounts

    feature_list = ['account2', 'amount', 'day', 'weekday']

    def get_mapped_features(self, data, acc_map):
        features = np.zeros((len(data), len(self.feature_list)))
        for i, row in enumerate(data):
            features[i][0] = acc_map[row['account2']]
            features[i][1] = row['amount']
            features[i][2] = row['day']
            features[i][3] = row['weekday']
        return features

    def get_transactions(self):
        return self.data[DATA]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Export gnucash transactions')

    parser.add_argument('--account', default=None, help="Filter by account")
    parser.add_argument('--use-cache', action='store_true')
    parser.add_argument('-v', '--invert', action='store_true', help="Invert keyword matching.")
    parser.add_argument('file_', help="gnucash file")
    parser.add_argument('keywords', nargs='*', help="List of required keywords")

    args = parser.parse_args()

    featureset = TransactionFeatureSet(args.file_, cache=args.use_cache)

    from assign_transactions import assignAccounts
    transactions = tuple(zip(
        featureset.data[DATA],
        featureset.full_features
    ))
    assignAccounts(
        transactions,
        featureset.full_classifier,
        featureset.data[ACCOUNT_LABELS],
    )
