import numpy as np
from .transaction import Transaction
from .transaction_set import TransactionSet
from .classifier import Classifier
from .common import word_inclusion_criteria
from .predict_ofx import get_txs

if __name__ == '__main__':
    args, txs, ofx_roots = get_txs()

    transaction_set = TransactionSet(txs, export_function=None)

    print(transaction_set.account_labels)
    print([tr.output_account for tr in transaction_set.assigned_transactions])

    classifier = Classifier(transaction_set=transaction_set)
    print([transaction_set.account_labels[i] for i in classifier.mapped_output_accounts])

    probabilities, predictions = classifier.predict(
        transaction_set.assigned_transactions
    )
    print(probabilities)
    print("Predicted (max prob) vs actual")
    print(predictions)
    print(np.array([np.argmax(probs) for probs in probabilities]))
    print(classifier.mapped_output_accounts)
