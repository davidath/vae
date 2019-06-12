import tensorflow as tf

# Create custom learning rate decay scheduler
# We want to decay the learning rate on specific epoch numbers

class decay_lr(tf.keras.callbacks.Callback):
    ''' 
        n_epoch = no. of epochs after decay should happen.
        decay = decay value
    '''  
    def __init__(self, n_epoch, decay):
        super(decay_lr, self).__init__()
        self.n_epoch=n_epoch
        self.decay=decay

    def on_epoch_begin(self, epoch, logs={}):
        old_lr = self.model.optimizer.lr
        old_lr = tf.keras.backend.get_value(old_lr)
        # If current epoch is divided by decay epoch then decay lr
        if epoch % self.n_epoch == 0 and epoch != 0:
            new_lr= self.decay*old_lr
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
        else:
            tf.keras.backend.set_value(self.model.optimizer.lr, old_lr)
