
class TransactionSet(object):
    def __init__(self, transactions, export_function):
        self.transactions = sorted(transactions, key=lambda tr:tr.fitid)
        self.export_function = export_function

    def flush_transactions(self):
        self.export_function()

    @property
    def account_labels(self):
        return sorted(tr.output_account for tr in self.assigned_transactions)

    @property
    def assigned_transactions(self):
        return [
            tr for tr in self.transactions
            if tr.output_account is not None
        ]

    @property
    def unassigned_transactions(self):
        return [
            tr for tr in self.transactions
            if tr not in self.assigned_transactions
        ]


