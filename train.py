from tensorflow.keras.utils import to_categorical
import extract_feats.librosa as lf
import models
from utils import parse_opt
import tensorflow as tf
from sklearn.utils import check_random_state
import time
import random
import numpy as np
n = 3
np.random.seed(n)
random.seed(n)
tf.random.set_seed(n)
check_random_state(n)

def train(config) -> None:

    # Load the entire dataset
    if config.feature_method == 'l':
        data = lf.load_feature(config, train=True)

    # Iterate over the folds
    for x_train, x_test, y_train, y_test in data:

        # train
        model = models.make(config=config, n_feats=x_train.shape[1])

        print('----- start training', config.model, '-----')

        # Note the time before training
        start_time = time.time()
        if config.model in ['PS_AC1D_FIF']:
            y_train, y_val = to_categorical(y_train), to_categorical(y_test)
            model.train(
                x_train, y_train,
                x_test, y_val,
                batch_size = config.batch_size,
                n_epochs = config.epochs
            )
        else:
            model.train(x_train, y_train)

        # Note the time after training
        print('----- end training ', config.model, ' -----')

        model.evaluate(x_test, y_test)
        end_time = time.time()

        # Calculate the difference
        training_time = end_time - start_time

        print(f"The training time of the model is: {training_time} seconds")
        model.save(config.checkpoint_path, config.checkpoint_name)


if __name__ == '__main__':
    config = parse_opt()
    train(config)
