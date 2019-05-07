import logging
logger = logging.getLogger(__name__)

from .common import word_inclusion_criteria

class Transaction(object):
    def __init__(
        self, full_description, base_account, output_account,
        amount, debit, date, day, isWeekend,
        tr_dict,
        fitid=None,
    ):
        self.full_description = full_description
        self.description=set(word.lower() for word in full_description.split() if word_inclusion_criteria(word))
        self.base_account = base_account
        self._output_account = output_account
        self.amount = amount
        self.debit = debit
        self.date = date
        self.day = day
        self.isWeekend = isWeekend
        self.fitid = fitid
        self.tr_dict = tr_dict

    @property
    def output_account(self):
        return self._output_account

    @output_account.setter
    def output_account(self, value):
        self._output_account = value
        if self.tr_dict:
            self.tr_dict['ACCTTO'] = value

    def __str__(self):
        return 'Debit: {} , Day: {}, Weekend {} , Amount {}\n{} -> {}\n'.format(
            self.debit,
            self.day,
            self.isWeekend,
            self.amount,
            self.base_account,
            self.output_account
        )

