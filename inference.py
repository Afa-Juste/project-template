import json
import pandas as pd
import numpy as np
from keras.models import load_model

class Inferencer(object):
    """We need this class in order to format the predictions as a table
    with the classes as strings instead of ints or one-hot vectors."""

    def __init__(self):
        self.data = None
        self.model = None
        self.class_names = None

    def load_data(self, datapath, class_names):
        self.data = np.load(datapath)
        self.class_names = class_names

    def load_model(self, modelpath):
        self.model = load_model(modelpath)

    def predict(self):
        predicted_labels = self.model.predict_classes(self.data, batch_size=32, verbose=0).tolist()
        predicted_probabilities = self.model.predict_proba(self.data, batch_size=32, verbose=0).tolist()

        string_predicted_labels = []

        for label in predicted_labels:
            string_predicted_labels.append(self.class_names[label])

        for index in range(len(predicted_probabilities)):
            predicted_probabilities[index] = np.amax(predicted_probabilities[index])

        df = pd.DataFrame({'Prediction':string_predicted_labels,
                'Confidence': predicted_probabilities})

        return df


def main():

    configs = json.load(open('inference_config.json', 'r'))

    inferencer = Inferencer()
    inferencer.load_data(configs['datapath'], configs['class_names'])
    inferencer.load_model(configs['modelpath'])

    print(inferencer.predict())


if __name__ == '__main__':
    main()
