# FCN model
# when tuning start with learning rate->mini_batch_size ->
# momentum-> #hidden_units -> # learning_rate_decay -> #layers
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time

from utils.utils import save_logs
from utils.utils import calculate_metrics


class callback_val_ASR(tf.keras.callbacks.Callback):
    def __init__(self, x_ASR, y_ASR, x_ASR_train, y_ASR_train):
        self.x_ASR = x_ASR
        self.y_ASR = y_ASR
        self.x_ASR_train = x_ASR_train
        self.y_ASR_train = y_ASR_train

    def on_epoch_end(self, epoch, logs=None):
        val_ASR = self.model.evaluate(self.x_ASR, self.y_ASR, verbose=0)
        val_ASR_train = self.model.evaluate(self.x_ASR_train, self.y_ASR_train, verbose=0)
        logs['ASR'] = val_ASR[1]
        print('ASR_test:', val_ASR[1])
        print('ASR_train:', val_ASR_train[1])


class Classifier_FCN:
    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True, c_loss=None):
        self.output_directory = output_directory
        if build == True:
            self.model = self.build_model(input_shape, nb_classes, c_loss)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory + 'model_init.hdf5')
        return

    def build_model(self, input_shape, nb_classes, c_loss=None):
        input_layer = keras.layers.Input(input_shape)

        conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation(activation='relu')(conv1)

        conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)

        conv3 = keras.layers.Conv1D(128, kernel_size=3, padding='same')(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)

        gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        if c_loss is None:
            model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                          metrics=['accuracy'])
        else:
            model.compile(loss=c_loss, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                      min_lr=0.0001)

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)

        self.callbacks = [reduce_lr, model_checkpoint]

        return model

    def _fit_model(self, x_train, y_train, x_val, y_val, eval_val_ASR, batch_size, nb_epochs):
        if not tf.test.is_gpu_available:
            print('error')
            exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training

        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val),
                              callbacks=self.callbacks + eval_val_ASR)

        duration = time.time() - start_time

        self.model.save(self.output_directory + 'last_model.hdf5')

        model = keras.models.load_model(self.output_directory + 'best_model.hdf5')

        return model, hist, duration

    def fit(self, x_train, y_train, x_val, y_val, y_test_classlabel, batch_size=16, nb_epochs=50):

        model, hist, duration = self._fit_model(x_train, y_train, x_val, y_val, [], batch_size, nb_epochs)

        y_pred = model.predict(x_val)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        save_logs(self.output_directory, hist, y_pred, y_test_classlabel, duration, calc_ASR=False)

        keras.backend.clear_session()

    def fit_backdoor(self, x_train, y_train, x_val, y_val,
                     pattern_generator, y_target=0, poison_rate=0.1,
                     clean_label=True, batch_size=16):
        y_val_classlabel = np.argmax(y_val, axis=1)

        x_ASR, y_ASR = pattern_generator(x_val, y_val, y_target, poison_rate=1.0,
                                         clean_label=False, one_hot=True, exclude_target=True)
        x_ASR_train, y_ASR_train = pattern_generator(x_train, y_train, y_target, poison_rate=1.0,
                                                     clean_label=False, one_hot=True, exclude_target=True)
        eval_val_ASR = callback_val_ASR(x_ASR, y_ASR, x_ASR_train, y_ASR_train)

        for e in range(500):
            x_train_backdoor, y_train_backdoor = pattern_generator(x_train, y_train, y_target,
                                                                   poison_rate=poison_rate, clean_label=clean_label,
                                                                   one_hot=True)
            print("Epoch:", e + 1)
            model, hist, duration = self._fit_model(x_train_backdoor, y_train_backdoor,
                                                    x_val, y_val, [eval_val_ASR], batch_size, nb_epochs=2)

        y_pred = model.predict(x_val)

        # convert the predicted from binary to integer
        y_pred_classlabel = np.argmax(y_pred, axis=1)

        # Backdoor attack
        x_val_backdoor, y_val_backdoor = pattern_generator(x_val, y_val, y_target,
                                                           poison_rate=1.0, clean_label=False, one_hot=False)
        y_pred_backdoor = np.argmax(model.predict(x_val_backdoor), axis=1)

        save_logs(self.output_directory, hist, y_pred_classlabel, y_val_classlabel, duration,
                  calc_ASR=True, y_pred_backdoor=y_pred_backdoor, y_target=y_val_backdoor)

        keras.backend.clear_session()
