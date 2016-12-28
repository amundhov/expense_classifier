import numpy as np
import piecash
import functools

from . import transaction

import logging
logger = logging.getLogger(__name__)

# from treeinterpreter import treeinterpreter as ti

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
# from treeinterpreter import treeinterpreter as ti
# from sklearn import metrics

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
        logging.info('{} -> {}'.format(*[tr.splits[i].account.fullname for i in [0,1]]))
        yield tr

def split(delimiters, string, maxsplit=0):
    import re
    regexPattern = '|'.join(map(re.escape, delimiters))
    return re.split(regexPattern, string, maxsplit)


def extract_transaction_features(tr, acc_map):
    delimiters = [" ", "/"]
    description = set(split(delimiters, tr.description))
    if tr.splits[0].account.fullname in PREDICTION_ACCOUNTS:
        base_account = tr.splits[0].account.fullname
        output_account = tr.splits[1].account.fullname
        debit = 0
    else:
        base_account = tr.splits[1].account.fullname
        output_account = tr.splits[0].account.fullname
        debit = 1
    assert base_account in PREDICTION_ACCOUNTS
    try:
        description.union(split(
            delimiters,
            tr.notes.split('OFX ext. info: |Memo:')[1]
        ))
    except AttributeError:
        pass
    except IndexError:
        description.union(split(
            delimiters,
            str(tr.notes)))
    return transaction.Transaction(
        full_description=tr.description,
        description=set(word for word in description if len(word) > 3),
        base_account=base_account,
        output_account=output_account,
        mapped_output_account=acc_map[output_account],
        amount=functools.reduce(lambda x,y: x or y, [spl.quantity for spl in tr.splits]),
        debit=debit,
        date=tr.post_date,
        day=tr.post_date.isoweekday(),
        isWeekend=int(tr.post_date.isoweekday() in [6, 7]),
    )

def get_word_mapping(table):
    words = list(set(
        word.lower()
        for row in table
        for word in row.description
    ))
    return {j:i for i, j in enumerate(words)}


DATA='data'
MAPPED_METADATA='mapped_metadata'
DATA='data'
MAPPED_METADATA='mapped_metadata'
MAPPED_DESCRIPTION='mapped_description'
ACCOUNT_LABELS='account_labels'
ACCOUNT_MAP='account_map'
WORD_MAP='word_map'
OUTPUT_ACCOUNTS='output_accounts'
MAPPED_OUTPUT_ACCOUNTS='mapped_output_accounts'

class TransactionFeatureSet(object):

    def __init__(self, book_file):
        self.book = piecash.open_book(book_file, readonly=True)

        self.account_labels = [a.fullname for a in self.book.accounts]
        self.acc_map = {j:i for i, j in enumerate(self.account_labels)}
        self.data = [
            extract_transaction_features(tr, self.acc_map)
            for tr in Prediction_Transactions(self.book)
        ]
        self.setup_classifier(self.data)

    def get_full_features(self, transactions):
        mapped_metadata = self.get_mapped_metadata(transactions, self.acc_map)
        mapped_description = self.get_mapped_description_features(transactions, self.word_map)
        description_features = self.get_description_probabilities(mapped_description)
        full_features = np.hstack([
            mapped_metadata,
            description_features,
        ])
        return full_features

    def setup_classifier(self, transactions):
        self.word_map = get_word_mapping(transactions)

        mapped_output_accounts = np.array([tr.mapped_output_account for tr in transactions], dtype=np.int)

        mapped_description = self.get_mapped_description_features(transactions, self.word_map)
        self.fit_descriptions(mapped_description, mapped_output_accounts)

        self.full_features = self.get_full_features(transactions)
        self.fit(self.full_features, mapped_output_accounts)


    def fit(self, full_features, output_classes):
        logger.info(full_features[0])
        self.full_classifier = RandomForestClassifier()
        self.full_classifier.fit(full_features, output_classes)
        #print("Score full classifier: {}".format(self.full_classifier.score(full_features, self.data[MAPPED_OUTPUT_ACCOUNTS])))
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
        logger.info('\n'.join(str(o) for o in zip(self.data[MAPPED_OUTPUT_ACCOUNTS], ['{} -> {}'.format(pred,out,) for pred, out in zip(predicted_labels, output_labels)])))

    def fit_descriptions(self, mapped_descriptions, mapped_accounts):
        self.description_model = GaussianNB()
        self.description_model.fit(mapped_descriptions, mapped_accounts)
        #print("Score bayesian description only: {}".format(self.description_model.score(self.data[MAPPED_DESCRIPTION], self.data[MAPPED_OUTPUT_ACCOUNTS])))

    def get_description_probabilities(self, descriptions):
        return np.array(self.description_model.predict_proba(descriptions))


    @staticmethod
    def get_mapped_description_features(data, word_map):
        features = np.zeros((len(data), len(word_map)))
        for i, row in enumerate(data):
            for word, j in word_map.items():
                features[i][j] = 1 if word in row.description else 0
        return features

    @staticmethod
    def get_output_accounts(data):
        mapped_output_accounts = np.array([tr.mapped_output_account for tr in data], dtype=np.int)
        #for i,  in enumerate(data):
        #    mapped_output_accounts[i] = row['mapped_output_account']
        #return output_accounts, mapped_output_accounts

    feature_list = ['base_account', 'amount', 'day', 'isWeekend']

    @classmethod
    def get_mapped_metadata(cls, transactions, acc_map):
        features = np.zeros((len(transactions), len(cls.feature_list)))
        for i, tr in enumerate(transactions):
            for j, feature in enumerate(cls.feature_list[1:]):
                features[i][j] = getattr(tr, feature)
            features[i][-1] = acc_map[tr.base_account]
        return features

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Export gnucash transactions')

    parser.add_argument('--account', default=None, help="Filter by account")
    parser.add_argument('-v', '--invert', action='store_true', help="Invert keyword matching.")
    parser.add_argument('file_', help="gnucash file")
    parser.add_argument('keywords', nargs='*', help="List of required keywords")

    args = parser.parse_args()

    featureset = TransactionFeatureSet(args.file_)

    from .assign_transactions import assignAccounts
    assignAccounts(
        featureset.data,
        featureset.full_features,
        featureset.full_classifier,
        featureset.account_labels
    )
