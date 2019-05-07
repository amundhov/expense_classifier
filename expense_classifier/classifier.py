
import numpy as np
import functools

import logging
logger = logging.getLogger(__name__)

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


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

def get_word_mapping(transactions):
    words = list(set(
        word.lower()
        for tr in transactions
        for word in tr.description
    ))
    return {j:i for i, j in enumerate(words)}

class Classifier(object):

    def __init__(self, transaction_set):
        self.transaction_set = transaction_set
        self.setup_classifier()

    def get_predictions(self, transaction):
        if not self.full_classifier:
            # No transactions to predict from
            return [('', 0)]

        mapped_features = self.get_full_features([transaction])
        logger.info(mapped_features)
        [probabilities] = self.full_classifier.predict_proba(mapped_features)
        logger.info(len(probabilities))
        logger.info(len(self.transaction_set.account_labels))
        logger.info(len(self.full_classifier))

        return [
            (self.transaction_set.account_labels[index], prob)
            for index, prob in enumerate(probabilities)
        ]

    def get_description_features(self, transactions):
        mapped_description = self.get_mapped_description_features(transactions, self.word_map)
        return self.get_description_probabilities(mapped_description)


    def get_full_features(self, transactions):
        mapped_metadata = self.get_mapped_metadata(
            transactions,
            self.transaction_set.account_labels
        )
        description_features = self.get_description_features(transactions)
        full_features = np.hstack([
            mapped_metadata,
            description_features,
        ])
        return full_features

    def setup_classifier(self):
        transactions = self.transaction_set.assigned_transactions
        if not transactions:
            self.full_classifier = None
            return

        self.word_map = get_word_mapping(transactions)
        self.mapped_output_accounts = np.array([self.transaction_set.account_labels.index(tr.output_account) for tr in transactions], dtype=np.int)

        mapped_description = self.get_mapped_description_features(transactions, self.word_map)
        self.fit_descriptions(mapped_description, self.mapped_output_accounts)

        full_features = self.get_full_features(transactions)

        self.full_classifier = RandomForestClassifier(n_estimators=30)
        self.full_classifier.fit(full_features, self.mapped_output_accounts)
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
        return mapped_output_accounts

    feature_list = ['base_account', 'amount', 'day', 'isWeekend']

    @classmethod
    def get_mapped_metadata(cls, transactions, account_list):
        features = np.zeros((len(transactions), len(cls.feature_list)))
        try:
            for i, tr in enumerate(transactions):
                for j, feature in enumerate(cls.feature_list[1:]):
                    features[i][j] = getattr(tr, feature)
                features[i][-1] = account_list.index(tr.base_account)
        except KeyError:
            logger.info(tr)
            raise

        return features

