import curses
from curses import wrapper
import urwid.curses_display
import urwid

class TransactionItem(urwid.Text):
    def __init__(self, text):
        super().__init__(text)
        urwid.register_signal(self.__class__, ['commit'])

    def keypress(self, size, key):
        if key == 'enter':
            urwid.emit_signal(self, 'commit')
        else:
            return key

    def selectable(self):
        return True

class TransactionBox(urwid.Pile):
    def __init__(self):
        self.date = urwid.Text('', align='right')
        self.amount = urwid.Text('', align='left')
        self.description = urwid.Text('')
        first_line = urwid.Columns([self.amount, self.date])
        super().__init__([first_line, self.description])
        self._loadTransaction(
            date='wed 17/03/2016',
            account='Assets::DNB::',
            amount='100,-',
            description='Foobar description',
        )

    def _loadTransaction(self, account, amount, date, description):
        self.date.set_text(date)
        self.amount.set_text('%s %s' % (account, amount))
        self.description.set_text(description)

    def loadTransaction(self, transaction):
        date = transaction['date'].strftime('%c')
        self._loadTransaction(
            account=transaction['account2'],
            amount='%s,-' % (transaction['amount']),
            date=date,
            description=' '.join(transaction['description']),
        )



class MainFrame(urwid.Frame):
    def __init__(self, transactions, classifier, account_labels):
        self.transactions = transactions
        self.classifier = classifier
        self.account_labels = account_labels

        self._index = 0
        self._last_index = 0
        self._listbox_items = []

        self.filter_edit = urwid.Edit(caption='filter: ')
        self.transaction_walker = urwid.SimpleFocusListWalker([])
        self.listbox = urwid.ListBox(self.transaction_walker)
        self.transaction_box = TransactionBox()
        super().__init__(
            urwid.LineBox(self.listbox),
            header= urwid.Pile([
                urwid.LineBox(self.transaction_box),
                self.filter_edit,
            ])
        )
        self.set_focus('body')

        if transactions is None:
            items = [
                urwid.AttrMap(
                    TransactionItem('transaction {}'.format(o)),
                    'selectable', 'focus')
                for o in range(0,10)
            ]
            self.transaction_walker.extend(items)
        else:
            self.loadTransaction(self._index)

    #def render(self, size, focus=False):
    #    if not self._last_index == self._index:
    #        self.loadTransaction(self._index)
    #    return super().render(size, focus=focus)

    def update(self):
        self.loadTransaction(self._index)

    def loadTransaction(self, index):
        transaction, mapped_features = self.transactions[index]
        self.transaction_box.loadTransaction(transaction)

        [probabilities] = self.classifier.predict_proba([mapped_features])
        predictions = [(index, prob) for index, prob in enumerate(probabilities)]
        predictions = sorted(predictions,
                             reverse=True,
                             key=lambda x: x[1],
                            )

        self.transaction_walker.clear()
        self.transaction_walker.extend([
            urwid.AttrMap(
                TransactionItem('{:.2%} {}'.format(
                    prob,
                    ':'.join(
                    reversed(
                        self.account_labels[
                            self.classifier.classes_[[output_class]]
                        ].split(':')
                    ))
                )),
                'selectable', 'focus'
            )
            for output_class, prob in predictions
            if prob > 0
        ])
        self.transaction_walker.append(urwid.Divider(
            div_char=u'-',
            top=1,
        ))
        self.transaction_walker.extend([
            urwid.AttrMap(
                TransactionItem('{:.2%} {}'.format(
                    prob,
                    ':'.join(
                    reversed(
                        self.account_labels[
                            self.classifier.classes_[[output_class]]
                        ].split(':')
                    ))
                )),
                'selectable', 'focus'
            )
            for output_class, prob in predictions
            if prob <= 0
        ])
        self.transaction_walker.set_focus(0)

    def keypress(self, size, key):
        if key in ['up', 'down']:
            self.listbox.keypress(size, key)
        elif key == 'right':
            if self._index < len(self.transactions):
                self._index += 1
                self.update()
        elif key == 'left':
            if self._index > 1:
                self._index -= 1
                self.update()
        else:
            self.filter_edit.keypress([size[0]], key)

def assignAccounts(transactions=None, classifier=None, account_labels=None,):
    palette = [
        ('focus', 'black,bold',  'light cyan'),
        #('selectable','black', 'dark cyan'),
    ]
    urwid.MainLoop(urwid.Padding(
        MainFrame(transactions, classifier, account_labels),
        align='center',
        width=('relative', 85),
    ), palette).run()

if __name__ == '__main__':
    assignAccounts()
