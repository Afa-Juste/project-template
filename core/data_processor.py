import random
from numpy import argmax, array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import pandas as pd

class DataProcessor(object):
    """DataProcessor converts string labels to one-hot encoded format
    and does a train/test split"""

    def __init__(self, data):
         self.data = pd.read_hdf(data, 'df')

    def process(self):

        # Load the values of each key into variables
        samples = self.data['samples'].values.tolist()
        labels = self.data['modulation'].values.tolist()
        snr = self.data['snr'].values.tolist()

        # 80/20 Train/Val/Test split. Also shuffles the values deterministically.
        samples_train, samples_test, labels_train, labels_test, snr_train,snr_test = train_test_split(samples,labels, snr, test_size=0.2, random_state=50)

        # Convert labels to one-hot encoding format
        label_encoder = LabelEncoder()
        labels_train = to_categorical(label_encoder.fit_transform(labels_train))
        labels_test = to_categorical(label_encoder.fit_transform(labels_test))

        #return samples_train, samples_test, labels_train, labels_test, snr_train, snr_test
        return array(samples_train), array(samples_test), array(labels_train), array(labels_test), array(snr_train), array(snr_test)


def main():
    pass

if __name__ == '__main__':
    main()
