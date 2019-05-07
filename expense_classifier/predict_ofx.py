import xmltodict
import datetime
from decimal import Decimal

from .transaction import Transaction
from .transaction_set import TransactionSet
from .classifier import Classifier
from .common import word_inclusion_criteria

import logging
logger = logging.getLogger(__name__)

FITID = 'FITID'
DTPOSTED = 'DTPOSTED'
NAME = 'NAME'
MEMO = 'MEMO'
ACCTTO = 'ACCTTO'
ACCTID = 'ACCTID'


def extract_transaction_features(tr, base_account):
    try:
        return _extract_transaction_features(tr, base_account)
    except:
        print(tr)
        raise

def _extract_transaction_features(tr, base_account):
    date = tr[DTPOSTED]
    name = tr[NAME] or '' if NAME in tr else ''
    memo = tr[MEMO] or '' if MEMO in tr else ''
    fitid = tr[FITID] if FITID in tr else None
    output_account = tr[ACCTTO] if ACCTTO in tr else None
    date = datetime.datetime(
        year=int(date[:4]),
        month=int(date[4:6:]),
        day=int(date[6:]),
    )
    description_words = []
    try:
        description_words.extend([o.strip() for o in name.split(' ')])
    except AttributeError:
        pass
    try:
        description_words.extend([o.strip() for o in memo.split(' ')])
    except AttributeError:
        pass

    if not (name or memo):
        full_description = 'Interest'
    else:
        full_description = ' / '.join([name, memo,])

    return Transaction(
        full_description=full_description,
        base_account=base_account,
        amount=Decimal(tr['TRNAMT']),
        debit=Decimal(tr['TRNAMT']) > 0,
        date=date,
        day=date.isoweekday(),
        isWeekend=int(date.isoweekday() in [6, 7]),
        output_account=output_account,
        fitid=fitid,
        tr_dict=tr,
    )

def get_account(ofx_root):
    return ofx_root['OFX']['BANKMSGSRSV1']['STMTTRNRS']['STMTRS']['BANKACCTFROM']['ACCTID']


def get_transactions(ofx_root):
    return ofx_root['OFX']['BANKMSGSRSV1']['STMTTRNRS']['STMTRS']['BANKTRANLIST']['STMTTRN']

def load_transactions(ofx_root):
    account_name = get_account(ofx_root)

    return [
       extract_transaction_features(
           tr, account_name
       )
       for tr in get_transactions(ofx_root)
       if tr['DTPOSTED'] is not None
    ]

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
    parser = argparse.ArgumentParser(description='Assign OFX transactions')
    parser.add_argument('--threshold', default=95, help="Percentage to assume transaction predicted.")
    parser.add_argument('paths', nargs='*', help="OFX files to process")

    args = parser.parse_args()


    ofx_roots = []
    for path in args.paths:
        with open(path, encoding='latin1') as file_:
            ofx_roots.append((
                xmltodict.parse(file_.read()),
                path
            ))

    txs=[
        tr
        for root, path in ofx_roots
        for tr in load_transactions(root)
    ]

    def export_transactions():
        logger.info('Unparsing updated transactions')
        for root, path in ofx_roots:
            with open(path, 'w') as file_:
                logger.info("Exporting to " + path)
                file_.write(xmltodict.unparse(root, pretty=True))

    transaction_set = TransactionSet(txs, export_function=export_transactions)
    classifier = Classifier(transaction_set=transaction_set)

    from .assign_transactions import assignAccounts
    assignAccounts(
        threshold=args.threshold,
        classifier=classifier,
    )

    logger.info('Assigned transaction outputs are now {}'.format(
        ' '.join(tr.output_account for tr in transaction_set.assigned_transactions)
    ))

