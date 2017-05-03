from keras.callbacks import Callback

class History(Callback):
    def on_train_begin(self,logs={}):
        self.tr_losses=[]
        self.val_losses=[]
        self.tr_accs=[]
        self.val_accs=[]

    def on_epoch_end(self,epoch,logs={}):
        self.tr_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.tr_accs.append(logs.get('acc'))
        self.val_accs.append(logs.get('val_acc'))

def dump_history(store_path,logs):
    with open('train_loss','a') as f:
        for loss in logs.tr_losses:
            f.write('{}\n'.format(loss))
    with open('train_accuracy','a') as f:
        for acc in logs.tr_accs:
            f.write('{}\n'.format(acc))
    with open('valid_loss','a') as f:
        for loss in logs.val_losses:
            f.write('{}\n'.format(loss))
    with open('valid_accuracy','a') as f:
        for acc in logs.val_accs:
            f.write('{}\n'.format(acc))
