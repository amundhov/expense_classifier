import xmltodict
import datetime
from decimal import Decimal

from . import transaction
from .common import word_inclusion_criteria

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
    name = tr['NAME'] or '' if 'NAME' in tr else ''
    memo = tr['MEMO'] or '' if 'MEMO' in tr else ''
    fitid = tr['FITID'] if 'FITID' in tr else None
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
    description_words = set(word.lower() for word in description_words if word_inclusion_criteria(word))
    if not (name or memo):
        full_description = 'Interest'
    else:
        full_description = ' / '.join([name, memo,])

    logger.info(description_words)

    return transaction.Transaction(
        full_description=full_description,
        description=description_words,
        base_account=base_account,
        amount=Decimal(tr['TRNAMT']),
        debit=Decimal(tr['TRNAMT']) > 0,
        date=date,
        day=date.isoweekday(),
        isWeekend=int(date.isoweekday() in [6, 7]),
        output_account=UNSPECIFIED_ACCOUNT,
        mapped_output_account=None,
        fitid=fitid,
    )

def get_account(ofx_dict):
    return ofx_dict['BANKACCTFROM']['ACCTID']

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
    parser = argparse.ArgumentParser(description='Import OFX transactions')
    parser.add_argument('--threshold', default=95, help="Percentage to assume transaction predicted.")
    parser.add_argument('path', help="OFX file")
    parser.add_argument('gnucash', help="Gnucash file")

    args = parser.parse_args()

    with open(args.path, encoding='latin1') as file_:
        ofx_dict = xmltodict.parse(file_.read())

    ofx_dict = ofx_dict['OFX']['BANKMSGSRSV1']['STMTTRNRS']['STMTRS']

    account = get_account(ofx_dict)

    transactions = [
       extract_transaction_features(tr, account)
	   for tr in  ofx_dict['BANKTRANLIST']['STMTTRN']
	   if tr['DTPOSTED'] is not None
    ]
    from .get_transactions import TransactionFeatureSet
    featureset = TransactionFeatureSet(args.gnucash)

	#logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

    from .assign_transactions import assignAccounts
    assignAccounts(
		threshold=args.threshold,
        transactions=transactions,
        feature_set=featureset,
    )
