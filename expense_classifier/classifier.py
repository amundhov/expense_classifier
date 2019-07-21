
import numpy as np
import functools

import logging
logger = logging.getLogger(__name__)

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


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

        [probabilities], predictions = self.predict([transaction])
        most_probable = np.argmax(probabilities)

        assert most_probable == predictions[0]
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
        self.mapped_output_accounts = np.array([self.transaction_set.account_labels.index(tr.output_account) for tr in transactions], dtype=np.int)
        if not transactions:
            self.full_classifier = None
            return

        self.description_vectorizer = CountVectorizer()
        X_train_counts = self.description_vectorizer.fit_transform(
                [tr.full_description for tr in transactions]
        )

        self.full_classifier = MultinomialNB().fit(
                X_train_counts,
                self.mapped_output_accounts
        )
        #self.full_classifier = RandomForestClassifier(n_estimators=10, random_state=0)
        #self.full_classifier.fit(full_features, self.mapped_output_accounts)

    def predict(self, transactions):
        X_counts = self.description_vectorizer.transform(
                [tr.full_description for tr in transactions]
        )
        logger.info(self.full_classifier.classes_)
        logger.info(self.transaction_set.account_labels)
        probabilities = np.zeros(
            (len(transactions), len(self.transaction_set.account_labels))
        )
        predictions = self.full_classifier.predict(X_counts)
        # n_transactions x n_classes_
        for i, log_probs in enumerate(
                self.full_classifier.predict_proba(X_counts)
        ):
            for j, prob in zip(self.full_classifier.classes_, log_probs):
                probabilities[i][j] = prob
        return (
            probabilities,
            predictions
        )
         

    def fit_descriptions(self, mapped_descriptions, mapped_accounts):
        self.description_model = GaussianNB()
        self.description_model.fit(mapped_descriptions, mapped_accounts)

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

