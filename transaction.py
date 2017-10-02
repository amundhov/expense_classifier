import logging
logger = logging.getLogger(__name__)

class Transaction(object):
    def __init__(
        self, full_description, description, base_account, output_account,
        mapped_output_account, amount, debit, date, day, isWeekend, fitid=None
    ):
        self.full_description = full_description
        self.description = description
        self.base_account = base_account
        self.output_account = output_account
        self.mapped_output_account = mapped_output_account
        self.amount = amount
        self.debit = debit
        self.date = date
        self.day = day
        self.isWeekend = isWeekend
        self.fitid = fitid

    def __str__(self):
        return 'Debit: {} , Day: {}, Weekend {} , Amount {}\n{} -> {}\n'.format(
            self.debit,
            self.day,
            self.isWeekend,
            self.amount,
            self.base_account,
            self.mapped_output_account
        )

