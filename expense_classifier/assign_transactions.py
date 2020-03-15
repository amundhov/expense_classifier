import sys
import urwid.curses_display
import urwid

from datetime import datetime

import logging
logger = logging.getLogger(__name__)


class TransactionItem(urwid.Text):
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
        self.progress = urwid.Text('(0 Assigned of 0)', align='right')
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
    def __init__(self, classifier, process_assigned_txs=False):
        self.transaction_set = classifier.transaction_set
        self.classifier = classifier

        self._index = 0
        self._listbox_items = []
        self.transactions = self.txs_to_assign

        self.filter_edit = urwid.Edit(caption='Filter: ')
        self.transaction_walker = urwid.SimpleFocusListWalker([])
        self.listbox = urwid.ListBox(self.transaction_walker)
        self.transaction_box = TransactionBox()
        super().__init__(
            body=urwid.LineBox(self.listbox),
            header=urwid.Pile([
                urwid.LineBox(self.transaction_box),
                self.filter_edit,
            ]),
            footer=urwid.Text(
'<LEFT> Previous | <TAB> <RIGHT> Next |  <ENTER> Refit | <F5> Flush to File | Type string to filter or create account',
align='center')
        )
        self.set_focus('body')

        if not self.transactions:
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

    @property
    def txs_to_assign(self):
        return sorted(self.transaction_set.unassigned_transactions, key=lambda x: x.date)

    def _transaction_line(self, output_account, probability):
        return urwid.AttrMap(
            TransactionItem(
                '{:.2%} {}'.format(
                    probability,
                    ':'.join(reversed(
                        output_account.split(':')
                    ))),
                output_account=output_account
            ),
            'selectable', 'focus'
        )

    def update(self):
        self.loadTransaction(self._index)

    def refit(self):
        self.classifier.setup_classifier()
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
        if output_account is None:
            output_account = self.filter_edit.edit_text.strip()
        logger.debug("Commiting current output %s", output_account)
        self.current_transaction.output_account = output_account

    def reset_filter(self):
        self.filter_edit.set_edit_text('')
        self.update_filter()

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
            lambda x: filter_string.lower() in x[0].lower(),
            self._predictions
        ))
        if len(predictions) == 0:
            self.filter_edit.set_caption(u'New account:\n')
        else:
            self.filter_edit.set_caption(u'Filter:\n')

        self._set_predictions(predictions)

        # Do linear-search for current focus in filtered list
        for local_index, (output, _) in enumerate(predictions):
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
        self.transaction_box.loadTransaction(transaction)

        self._predictions = self.classifier.get_predictions(transaction)
        self._predictions = sorted(
            s lf._predictions,
            reverse=True,
            key=lambda x: x[1],
        )
        self._set_predictions(self._predictions)

        if transaction.output_account is not None:
            try:
                focus = [
                    output_account for output_account, _ in self._predictions
                ].index(transaction.output_account)
            except ValueError:
                focus = 0
        else:
            focus = 0
        self.transaction_walker.set_focus(focus)

    def keypress(self, size, key):
        if key in ['up', 'down', 'page down', 'page up']:
            logger.info("Sending %s %s", key, " to listbox")
            self.listbox.keypress(size, key)
        elif key == 'enter':
            self.commit()
            self.refit()
        elif key == '<F5>':
            self.transaction_set.flush_transactions()
        elif key == 'right' or key == 'tab':
            self.commit()
            self.reset_filter()
            if self._index < len(self.transactions)-1:
                self._index += 1
                self.update()
                # Check if probability is over threshold. Recurse.
                #if self.current_transaction
        elif key == 'left':
            if self._index >= 1:
                self._index -= 1
                self.update()
        elif key == 'esc':
            logger.info("Exiting mainloop")
            self.transaction_set.flush_transactions()
            raise urwid.ExitMainLoop()

        else:
            self.filter_edit.keypress([size[0]], key)
            self.update_filter()


def assignAccounts(threshold, classifier):
    palette = [
        ('focus', 'black,bold',  'light cyan'),
    ]
    urwid.MainLoop(urwid.Padding(
        MainFrame(classifier),
        align='center',
        width=('relative', 85),
    ), palette).run()

if __name__ == '__main__':
    assignAccounts()
