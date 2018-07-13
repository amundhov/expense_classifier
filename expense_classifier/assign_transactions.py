import sys
import urwid.curses_display
import urwid

from datetime import datetime

from decimal import Decimal
from piecash import Transaction, Split

import logging
logger = logging.getLogger(__name__)


class TransactionItem(urwid.Text):
    # TODO keep output account to enable searching index for correct index
    def __init__(self, text, output_account=None):
        super().__init__(text)
        self.output_account = output_account

    def selectable(self):
        return True

    def keypress(self, size, key):
        return key

class TransactionBox(urwid.Pile):
    def __init__(self):
        self.date = urwid.Text('', align='right')
        self.amount = urwid.Text('', align='left')
        self.description = urwid.Text('')
        self.progress = urwid.Text('(0/0)', align='right')
        first_line = urwid.Columns([self.amount, self.date])
        second_line = urwid.Columns([self.description, self.progress])
        super().__init__([first_line, second_line])

        # Load default transaction
        self._loadTransaction(
            date='',
            account='',
            amount='',
            description='No new transactions found',
        )

    def _loadTransaction(self, account, amount, date, description):
        self.date.set_text(date)
        self.amount.set_text('%s %s' % (account, amount))
        self.description.set_text(description)

    def loadTransaction(self, transaction):
        date = transaction.date.strftime('%c')
        self._loadTransaction(
            account=transaction.base_account,
            amount='%s %s,-' % (
                '-' if transaction.debit else '',
                transaction.amount
            ),
            date=date,
            description=transaction.full_description,
        )


class MainFrame(urwid.Frame):
    def __init__(self, transactions, feature_set):
        existing_fitids = set(
                tr.fitid
                for tr in feature_set.data
        ).difference({None})
        self.transactions = [ tr for tr in transactions if tr.fitid not in existing_fitids ]
        self.transactions = sorted(self.transactions, key=lambda x: x.date)
        self.account_labels = feature_set.account_labels
        self.feature_set = feature_set

        self._index = 0
        self._listbox_items = []

        self.filter_edit = urwid.Edit(caption='filter: ')
        self.transaction_walker = urwid.SimpleFocusListWalker([])
        self.listbox = urwid.ListBox(self.transaction_walker)
        self.transaction_box = TransactionBox()
        super().__init__(
            urwid.LineBox(self.listbox),
            header=urwid.Pile([
                urwid.LineBox(self.transaction_box),
                self.filter_edit,
            ])
        )
        self.set_focus('body')

        if transactions is None:
            items = [
                urwid.AttrMap(
                    TransactionItem(
                        'transaction {}'.format(o)
                    ),
                    'selectable', 'focus')
                for o in range(0, 10)
            ]
            self.transaction_walker.extend(items)
        else:
            self.loadTransaction(self._index)

    def _transaction_line(self, output_account, probability):
        return urwid.AttrMap(
            TransactionItem(
                '{:.2%} {}'.format(
                    probability,
                    ':'.join(reversed(
                        self.account_labels[output_account].split(':')
                    ))),
                output_account=output_account
            ),
            'selectable', 'focus'
        )

    def update(self):
        self.loadTransaction(self._index)

    def refit(self):
        self.feature_set.fit()
        self.update()

    @property
    def current_transaction(self):
        return self.transactions[self._index]

    @property
    def current_output(self):
        try:
            current_item = self.transaction_walker.get_focus()[0].original_widget
            current_output = current_item.output_account
        except AttributeError:
            current_output = None
        return current_output

    def commit(self):
        output_account = self.current_output
        self.current_transaction.mapped_output_account = output_account
        self.current_transaction.output_account = self.account_labels[output_account]
        self.filter_edit.set_edit_text('')

    def save_transactions(self):
        commited_transactions = [
            tran
            for tran in self.transactions
            if tran.mapped_output_account is not None
        ]
        for tr in commited_transactions:
            notes = 'OFX ext. info: |FITID:{}'.format(tr.fitid) if tr.fitid else ''
            from_account = self.feature_set.gc_accounts[tr.base_account]
            to_account = self.feature_set.gc_accounts[tr.output_account]
            value=Decimal(tr.amount)
            currency=from_account.commodity

            logger.info(from_account)
            logger.info(to_account)

            gc_tran = Transaction(
                post_date=tr.date,
                enter_date=tr.date,
                currency=currency,
                notes=notes,
                description=tr.full_description,
                splits = [
                    Split(account=from_account, value=value),
                    Split(account=to_account, value=-value),
                ]
            )
            logger.info(gc_tran)
            gc_tran.validate()

        self.feature_set.book.save()
        for tr in commited_transactions:
            self.transactions.remove(tr)

        self._index = 0
        self.update()

    def _set_predictions(self, predictions):
        self.transaction_walker.clear()
        self.transaction_walker.extend([
            self._transaction_line(output_account, prob)
            for output_account, prob in predictions
        ])

    def update_filter(self):

        filter_string = self.filter_edit.edit_text.strip().lower()
        if len(filter_string) == 0:
            self._set_predictions(self._predictions)
        self.transaction_walker.clear()
        predictions = list(filter(
            lambda x: filter_string in self.account_labels[x[0]].lower(),
            self._predictions
        ))
        self._set_predictions(predictions)

        # Do linear-search for current focus in filtered list
        #logger.info(predictions)
        #logger.info('Searching for  {}'.format(current_output))
        for local_index, (output, _) in enumerate(predictions):
            #logger.info('{} {}'.format(local_index, output))
            if output == self.current_output:
                self.transaction_walker.set_focus(local_index)
                return

        self.transaction_walker.set_focus(0)

    def loadTransaction(self, index):
        if len(self.transactions) == 0:
            return

        self.transaction_box.progress.set_text('({}/{})'.format(
            index+1,
            len(self.transactions)
        ))

        transaction = self.transactions[index]
        logger.info(str(transaction))
        self.transaction_box.loadTransaction(transaction)

        mapped_features = (self.feature_set.get_full_features(self.transactions))[index]
        [probabilities] = self.feature_set.full_classifier.predict_proba([mapped_features])
        logger.info(len(probabilities))
        logger.info(len(self.feature_set.account_labels))
        logger.info(len(self.feature_set.full_classifier.classes_))
        logger.info(self.transactions[index].full_description)

        self._predictions = [
            (self.feature_set.full_classifier.classes_[index], prob)
            for index, prob in enumerate(probabilities)
        ]
        self._predictions = sorted(
            self._predictions,
            reverse=True,
            key=lambda x: x[1],
        )
        self._predictions.extend(
            (index, 0)
            for account_label, index in self.feature_set.acc_map.items()
            if index not in self.feature_set.full_classifier.classes_
        )
        self._set_predictions(self._predictions)

        if transaction.output_account is not None:
            try:
                focus = [
                    output_account for output_account, _ in self._predictions
                ].index(transaction.mapped_output_account)
            except ValueError:
                focus = 0
        else:
            focus = 0
        self.transaction_walker.set_focus(focus)

    def keypress(self, size, key):
        if key in ['up', 'down', 'page down', 'page up']:
            self.listbox.keypress(size, key)
        elif key == 'right' or key == 'tab':
            self.commit()

            if self._index < len(self.transactions)-1:
                self._index += 1
                self.update()
                # Check if probability is over threshold. Recurse.
                #logger.info()
                #if self.current_transaction
            else:
                # No more transactions save and exit
                self.save_transactions()
                logger.info("Saved all Transactions to Gnucash")
                raise urwid.ExitMainLoop

        elif key == 'left':
            self.commit()
            if self._index >= 1:
                self._index -= 1
                self.update()
        elif key == 'enter':
            self.save_transactions()
        elif key == 'F5':
            self.refit()
        else:
            self.filter_edit.keypress([size[0]], key)
            self.update_filter()


def assignAccounts(threshold, transactions=None, feature_set=None):
    palette = [
        ('focus', 'black,bold',  'light cyan'),
    ]
    urwid.MainLoop(urwid.Padding(
        MainFrame(transactions, feature_set),
        align='center',
        width=('relative', 85),
    ), palette).run()

if __name__ == '__main__':
    assignAccounts()
