import numpy as np
import piecash
import functools

from . import transaction
from .common import word_inclusion_criteria

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
            # Only want transactions that we have imported ourselves, in order
            # to avoid duplicates. We put unique id in notes.
            continue
        if len(tr.splits) != 2:
            # We don't do multi-split transactions.
            continue
        if not any(
            split.account.fullname in PREDICTION_ACCOUNTS
            for split in tr.splits
        ):
            continue
        yield tr

def Prediction_Transactions(book):
    for tr in book.transactions:
        #if tr.notes and 'OFX' in tr.notes:
        #    logger.info(tr.notes)
        if len(tr.splits) != 2:
            continue
        if not any(
            split.account.fullname in PREDICTION_ACCOUNTS
            for split in tr.splits
        ):
            continue
        #logging.info('{} -> {}'.format(*[tr.splits[i].account.fullname for i in [0,1]]))
        yield tr

def split(delimiters, string, maxsplit=0):
    import re
    regexPattern = '|'.join(map(re.escape, delimiters))
    return re.split(regexPattern, string, maxsplit)


def extract_transaction_features(tr, acc_map):
    delimiters = [" ", "/"]
    if 'Dato' in tr.description:
        tr.description = tr.description.split('Dato')[0]
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
    fitid=None
    try:
        if 'OFX ext. info:' in tr.notes:
            if '|Memo:' in tr.notes:
                description.union(split(
                    delimiters,
                    tr.notes.split('|Memo:')[1]
                ))
            if '|FITID:' in tr.notes:
                fitid = tr.notes.split('|FITID:')[1].split('|')[0]
                logger.debug("Found FITID {}".format(fitid))
    except (AttributeError, TypeError):
        pass
    except IndexError:
        logger.info(tr.notes)
        description.union(split(
            delimiters,
            str(tr.notes)))
    return transaction.Transaction(
        full_description=tr.description,
        description=set(word.lower() for word in description if word_inclusion_criteria(word)),
        base_account=base_account,
        output_account=output_account,
        mapped_output_account=acc_map[output_account],
        amount=functools.reduce(lambda x,y: x or y, [spl.quantity for spl in tr.splits]),
        debit=debit,
        date=tr.post_date,
        day=tr.post_date.isoweekday(),
        isWeekend=int(tr.post_date.isoweekday() in [6, 7]),
        fitid=fitid,
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
        self.book = piecash.open_book(book_file, readonly=False)

        self.account_codes = {a.code: a.fullname for a in self.book.accounts if a.code}
        self.account_labels = [a.fullname for a in self.book.accounts]
        self.acc_map = {j:i for i, j in enumerate(self.account_labels)}
        self.data = [
            extract_transaction_features(tr, self.acc_map)
            for tr in Prediction_Transactions(self.book)
        ]
        self.setup_classifier(self.data)

        self.gc_accounts = {
                account.fullname : account
                for account in self.book.accounts
        }
        logger.info(self.acc_map)
            

    def get_description_features(self, transactions):
        mapped_description = self.get_mapped_description_features(transactions, self.word_map)
        return self.get_description_probabilities(mapped_description)
        

    def get_full_features(self, transactions):
        mapped_metadata = self.get_mapped_metadata(transactions, self.acc_map)
        description_features = self.get_description_features(transactions)
        full_features = np.hstack([
            mapped_metadata,
            description_features,
        ])
        return full_features

    def setup_classifier(self, transactions):
        self.word_map = get_word_mapping(transactions)

        self.mapped_output_accounts = np.array([tr.mapped_output_account for tr in transactions], dtype=np.int)

        mapped_description = self.get_mapped_description_features(transactions, self.word_map)
        self.fit_descriptions(mapped_description, self.mapped_output_accounts)

        self.full_features = self.get_full_features(transactions)
        self.fit()


    def fit(self):
        self.full_classifier = RandomForestClassifier(n_estimators=30)
        self.full_classifier.fit(self.full_features, self.mapped_output_accounts)
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

    feature_list = ['base_account', 'amount', 'day', 'isWeekend']

    @classmethod
    def get_mapped_metadata(cls, transactions, acc_map):
        features = np.zeros((len(transactions), len(cls.feature_list)))
        try:
            for i, tr in enumerate(transactions):
                for j, feature in enumerate(cls.feature_list[1:]):
                    features[i][j] = getattr(tr, feature)
                features[i][-1] = acc_map[tr.base_account]
        except KeyError:
            logger.info(tr)
            raise

        return features

if __name__ == '__main__':
    import logging.config
    from .logging_config import config
    logging.config.dictConfig(config)
    logging.captureWarnings(True)

    import numpy
    def err_handler(type_, flag):
        logger.warning("Floating point error (%s), with flag %s" % (type, flag))
    numpy.seterrcall(err_handler)
    numpy.seterr(all='call')

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
        featureset
    )
