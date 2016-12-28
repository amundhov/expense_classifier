import xmltodict
import datetime

from . import transaction

import logging
logger = logging.getLogger(__name__)


UNSPECIFIED_ACCOUNT = 'Unspecified:Unmatched'

def extract_transaction_features(tr, base_account):
    try:
        return _extract_transaction_features(tr, base_account)
    except:
        print(tr)
        raise

def _extract_transaction_features(tr, base_account):
    date = tr['DTPOSTED']
    name = tr['NAME'] or ''
    memo = tr['MEMO'] or ''
    date = datetime.date(
        year=int(date[:4]),
        month=int(date[4:6:]),
        day=int(date[6:]),
    )
    description_words = []
    try:
        description_words.extend([o.strip() for o in tr['NAME'].split(' ')])
    except AttributeError:
        pass
    try:
        description_words.extend([o.strip() for o in tr['MEMO'].split(' ')])
    except AttributeError:
        pass
    description_words = set(word for word in description_words if len(word) > 3)
    if name not in memo:
        full_description = ' / '.join([name, memo,])
    else:
        full_description = name

    return transaction.Transaction(
        full_description=full_description,
        description=description_words,
        base_account=base_account,
        amount=float(tr['TRNAMT']),
        debit=float(tr['TRNAMT']) > 0,
        date=date,
        day=date.isoweekday(),
        isWeekend=int(date.isoweekday() in [6, 7]),
        output_account=UNSPECIFIED_ACCOUNT,
        mapped_output_account=None,
    )

def get_account(ofx_dict):
    return ofx_dict['BANKACCTFROM']['ACCTID']

if __name__ == '__main__':
    import logging.config
    from .logging_config import config
    logging.config.dictConfig(config)

    import argparse
    parser = argparse.ArgumentParser(description='Import OFX transactions')
    parser.add_argument('path', help="OFX file")
    parser.add_argument('gnucash', help="Gnucash file")

    args = parser.parse_args()

    with open(args.path, encoding='latin1') as file_:
        ofx_dict = xmltodict.parse(file_.read())

    ofx_dict = ofx_dict['OFX']['BANKMSGSRSV1']['STMTTRNRS']['STMTRS']

    account = get_account(ofx_dict)

    transactions = [
       extract_transaction_features(tr, account) for tr in  ofx_dict['BANKTRANLIST']['STMTTRN']
    ]
    from .get_transactions import TransactionFeatureSet
    featureset = TransactionFeatureSet(args.gnucash)

    from .assign_transactions import assignAccounts
    assignAccounts(
        transactions,
        featureset.get_full_features(transactions),
        featureset.full_classifier,
        featureset.account_labels,
    )
