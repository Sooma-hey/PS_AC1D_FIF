from keras.src.optimizers.legacy.rmsprop import RMSProp
from tensorflow.keras.layers import  Dense, Dropout, Flatten, Conv1D, Activation, BatchNormalization, MaxPooling1D, Input
from tensorflow.keras.models import Sequential, Model
from .dnn import DNN
import random
import numpy as np
import tensorflow as tf
from sklearn.utils import check_random_state
n = 3
np.random.seed(n)
random.seed(n)
tf.random.set_seed(n)
check_random_state(n)

class CNN1D(DNN):
    def __init__(self, model: Sequential, trained: bool = False) -> None:
        super(CNN1D, self).__init__(model, trained)


    @classmethod
    def make(
        cls,
        input_shape: int,
        n_classes: int = 6, # 6 for EMODB, 7 for RAVDESS
        lr: float = 0.001,
    ):


        def model_type(model):

            if model == 'PS_AC1D_FIF':

                input_layer = Input(shape=(input_shape, 1))

                #the convolutional layers
                conv1 = Conv1D(filters=60, kernel_size=(10), padding="same", activation="relu")(
                    input_layer)
                batch_norm0 = BatchNormalization()(conv1)
                conv2 = Conv1D(filters=80, kernel_size=(10), padding="same", activation="relu")(batch_norm0)
                drop0 = Dropout(0.1)(conv2)
                batch_norm1 = BatchNormalization()(drop0)
                conv3 = Conv1D(filters=100, kernel_size=(10), padding="same", activation="relu")(batch_norm1)
                batch_norm2 = BatchNormalization()(conv3)
                conv4 = Conv1D(filters=100, kernel_size=(10), padding="same", activation="relu")(batch_norm2)
                drop1 = Dropout(0.5)(conv4)
                batch_norm3 = BatchNormalization()(drop1)
                flatten = Flatten()(batch_norm3)

                # Add the output layer
                output_layer = Dense(n_classes, activation='softmax')(flatten)

                # Create the model
                model = Model(inputs=input_layer, outputs=output_layer)
                return model


        model = model_type('PS_AC1D_FIF')
        optimzer =RMSProp(learning_rate=lr, decay=1e-6)
        model.compile(loss='categorical_crossentropy', optimizer=optimzer, metrics=['accuracy'])

        return cls(model)

    def reshape_input(self, data: np.ndarray) -> np.ndarray:

        data = np.reshape(data, (data.shape[0], data.shape[1], 1))
        return data
