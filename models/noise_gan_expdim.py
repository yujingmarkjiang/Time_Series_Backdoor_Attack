# Noise-GAN model
import tensorflow.keras as keras
from keras import backend as K
import tensorflow as tf
import numpy as np
import time

from utils.utils import save_logs
from utils.utils import calculate_metrics

from models.fcn import callback_val_ASR


class Classifier_Noise_GAN:
    def __init__(self, output_directory, input_shape, verbose=False, build=True, c_loss=None):
        self.output_directory = output_directory
        if build == True:
            self.model = self.build_model(input_shape, c_loss)
            self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory + 'generator_init.hdf5')
        return

    def build_model(self, input_shape, c_loss=None):
        input_layer = keras.layers.Input(tuple(list(input_shape) + [1]))

        conv1 = keras.layers.Conv1D(filters=128*input_shape[1], kernel_size=15, padding='same', name='conv1')(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation(activation='relu')(conv1)

        conv2 = keras.layers.Conv1D(filters=512*input_shape[1], kernel_size=21, padding='same', name='conv2')(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)

        conv3 = keras.layers.Conv1D(filters=512, kernel_size=15, padding='same', name='conv3')(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)

        fc1 = keras.layers.Dense(256, activation='relu')(conv3)
        fc2 = keras.layers.Dense(1, activation='relu')(fc1)
        output_layer = fc2
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        return model

    def clip_add(self, pattern, ori_data):
        return (1 + pattern* 0.2) * ori_data

    def get_full_model(self, backdoor_clf, gen_trainable=True, bd_trainable=True):
        final_out = backdoor_clf.model(self.clip_add(self.model.outputs[0], self.model.inputs[0]))
        full_model = keras.models.Model(inputs=self.model.input, outputs=final_out)
        backdoor_clf.model.trainable = bd_trainable
        self.model.trainable = gen_trainable
        full_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                           metrics=['accuracy'])
        return full_model

    def _fit_backdoor(self, backdoor_clf, x_train, y_train, x_test, y_test, y_target, y_test_classlabel,
                      poison_rate, clean_label):
        x_test_backdoor, y_test_backdoor = self.process_instances(x_test, y_test, y_target,
                                                                  poison_rate=1.0, clean_label=clean_label,
                                                                  one_hot=True)
        x_train_backdoor_f, y_train_backdoor_f = self.process_instances(x_train, y_train, y_target,
                                                                        poison_rate=1.0, clean_label=clean_label,
                                                                        one_hot=True)

        print(backdoor_clf.model.evaluate(x_test, y_test)[1])

        x_train_backdoor, y_train_backdoor = self.process_instances(x_train, y_train, y_target,
                                                                    poison_rate, clean_label, one_hot=True,
                                                                    only_target=True)
        for e in range(1, 40):
            print("Epoch:", e)
            '''
            for layer in backdoor_clf.model.layers:
                layer.trainable = False
            '''

            # Train noise generator
            full_model = self.get_full_model(backdoor_clf, True, False)
            #K.set_value(full_model.optimizer.learning_rate, 0.01)

            full_model.fit(x_train_backdoor, y_train_backdoor, batch_size=4, epochs=10)

            # Train backdoor classifier
            full_model = self.get_full_model(backdoor_clf, False, True)

            full_model.fit(x_train_backdoor, y_train_backdoor, batch_size=16, epochs=10)

            val_clean_acc = backdoor_clf.model.evaluate(x_test, y_test)[1]
            val_ASR = full_model.evaluate(x_test_backdoor, y_test_backdoor)[1]
            val_ASR_train = full_model.evaluate(x_train_backdoor_f, y_train_backdoor_f)[1]
            print('!' * 10 + ' In Progress ' + '!' * 10)
            print("Clean acc.:", val_clean_acc)
            print("ASR:", val_ASR)
            print("ASR_train:", val_ASR_train)
            print('!' * 10 + ' In Progress ' + '!' * 10)

            #K.set_value(backdoor_clf.model.optimizer.learning_rate, 0.002)
            backdoor_clf.fit(x_train, y_train, x_test, y_test, y_test_classlabel, nb_epochs=int(4*e**0.3))

            val_clean_acc = backdoor_clf.model.evaluate(x_test, y_test)[1]
            val_ASR = full_model.evaluate(x_test_backdoor, y_test_backdoor)[1]
            val_ASR_train = full_model.evaluate(x_train_backdoor_f, y_train_backdoor_f)[1]
            print('#' * 20)
            print("Clean acc.:", val_clean_acc)
            print("ASR:", val_ASR)
            print("ASR_train:", val_ASR_train)
            print('#' * 20)
            self.model.save_weights(self.output_directory + 'generator_save/' + \
                                    f'epoch{e}_{val_clean_acc:.3f}_{val_ASR:.3f}_{val_ASR_train:.3f}.hdf5')
            backdoor_clf.model.save_weights(self.output_directory + 'backdoor_save/' + \
                                            f'epoch{e}_{val_clean_acc:.3f}_{val_ASR:.3f}_{val_ASR_train:.3f}.hdf5')

        return

    def fit(self, x_train, y_train, x_test, y_test, y_test_classlabel, backdoor_clf, process_instances,
            y_target=0, poison_rate=0.1, clean_label=False):
        self.process_instances = process_instances

        print("Pre-training...")
        backdoor_clf.fit(x_train, y_train, x_test, y_test, y_test_classlabel, nb_epochs=100)
        self._fit_backdoor(backdoor_clf, x_train, y_train, x_test, y_test, y_target, y_test_classlabel,
                           poison_rate, clean_label)
        self.model.save_weights(self.output_directory + 'generator_final.hdf5')
        backdoor_clf.model.save_weights(self.output_directory + 'backdoor_final.hdf5')

        return
