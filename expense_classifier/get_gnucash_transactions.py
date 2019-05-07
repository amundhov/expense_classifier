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
    'Assets:Nordea:Lonnskonto',
    'Assets:Nordea:Regningskonto',
    'Assets:Nordea:Huskonto',
    'Liabilities:Credit Cards:DNB Mastercard',
    'Liabilities:Credit Cards:Bank Norwegian',
]

#        gc_accounts = {
#                account.fullname : account
#                for account in self.book.accounts
#        }
#
#        logger.info("Saved all Transactions to Gnucash")
#
#        for tr in commited_transactions:
#            notes = 'OFX ext. info: |FITID:{}'.format(tr.fitid) if tr.fitid else ''
#            from_account = self.feature_set.gc_accounts[tr.base_account]
#            to_account = self.feature_set.gc_accounts[tr.output_account]
#            value=Decimal(tr.amount)
#            currency=from_account.commodity
#
#            logger.info(from_account)
#            logger.info(to_account)
#
#            gc_tran = piecash.Transaction(
#                post_date=tr.date,
#                enter_date=tr.date,
#                currency=currency,
#                notes=notes,
#                description=tr.full_description,
#                splits = [
#                    piecash.Split(account=from_account, value=value),
#                    piecash.Split(account=to_account, value=-value),
#                ]
#            )
#            logger.info(gc_tran)
#            gc_tran.validate()
#
#        self.gnucash_book.save()


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


def split(delimiters, string, maxsplit=0):
    import re
    regexPattern = '|'.join(map(re.escape, delimiters))
    return re.split(regexPattern, string, maxsplit)


def get_gnucash_transactions(self, book_file):
    book = piecash.open_book(book_file, readonly=False)
    account_labels = [a.fullname for a in self.book.accounts]
    acc_map = {j:i for i, j in enumerate(self.account_labels)}

    transactions = [
	extract_transaction_features(tr, acc_map)
	for tr in Prediction_Transactions(self.book)
        ]
    return  transactions

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


@staticmethod
def extract_gnucash_transactions(tr, acc_map):
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

    transactions = GnucashTransctions(args.file_)
    featureset = Classifier(transactions)

    from .assign_transactions import assignAccounts
    assignAccounts(
        featureset.data,
        featureset
    )
