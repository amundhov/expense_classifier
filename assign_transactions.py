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
            account=transaction['base_account'],
            amount='%s%s,-' % (
                '-' if transaction['debit'] else '',
                transaction['amount']
            ),
            date=date,
            description=transaction['full_description'],
        )



class MainFrame(urwid.Frame):
    def __init__(self, transactions, full_features, classifier, account_labels):
        self.transactions = transactions
        self.full_features = full_features
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

    def _transaction_line(self, output_account, probability):
        return urwid.AttrMap(
            TransactionItem('{:.2%} {}'.format(
                probability,
                ':'.join(
                reversed(
                    self.account_labels[output_account].split(':')
                ))
            )),
            'selectable', 'focus'
        )

    def update(self):
        self.loadTransaction(self._index)

    def refit(self):
        pass

    def commit(self):
        walker_index = self.transaction_walker.focus
        output_account, _ = self._predictions[walker_index]
        self.transactions[self._index]['mapped_output_account'] = output_account
        self.transactions[self._index]['output_account'] = self.account_labels[output_account]

    def loadTransaction(self, index):
        transaction = self.transactions[index]
        mapped_features = self.full_features[index]
        self.transaction_box.loadTransaction(transaction)

        [probabilities] = self.classifier.predict_proba([mapped_features])
        self._predictions = [
            (self.classifier.classes_[[index]], prob) 
            for index, prob in enumerate(probabilities)
        ]
        self._predictions = sorted(self._predictions,
                             reverse=True,
                             key=lambda x: x[1],
                            )
        self.transaction_walker.clear()
        self.transaction_walker.extend([
            self._transaction_line(output_account, prob)
            for output_account, prob in self._predictions
            if prob > 0
        ])
        self.transaction_walker.append(urwid.Divider(
            div_char=u'-',
            top=1,
        ))
        self.transaction_walker.extend([
            self._transaction_line(output_account, prob)
            for output_account, prob in self._predictions
            if prob <= 0
        ])

        if 'output_account' in transaction:
            try:
                focus = [
                    output_account for output_account, _ in self._predictions
                ].index(transaction['mapped_output_account'])
            except ValueError:
                focus = 0
        else:
            focus = 0
        self.transaction_walker.set_focus(focus)


    def keypress(self, size, key):
        if key in ['up', 'down']:
            self.listbox.keypress(size, key)
        elif key == 'right':
            self.commit()
            if self._index < len(self.transactions):
                self._index += 1
                self.update()
        elif key == 'left':
            self.commit()
            if self._index >= 1:
                self._index -= 1
                self.update()
        else:
            self.filter_edit.keypress([size[0]], key)

def assignAccounts(transactions=None, full_features=None, classifier=None, account_labels=None,):
    palette = [
        ('focus', 'black,bold',  'light cyan'),
        #('selectable','black', 'dark cyan'),
    ]
    urwid.MainLoop(urwid.Padding(
        MainFrame(transactions, full_features, classifier, account_labels),
        align='center',
        width=('relative', 85),
    ), palette).run()

if __name__ == '__main__':
    assignAccounts()
