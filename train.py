import os
import shutil
import json
from keras.models import Sequential
from keras.layers import Reshape, Dense, Flatten
from keras.optimizers import adam
from core.data_processor import DataProcessor
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
import mlflow.keras
import mlflow

def main():
    configs = json.load(open('train_config.json', 'r'))
    mlflow.set_experiment(configs['experiment_name'])

    with mlflow.start_run():

        # Set MLFlow logs
        mlflow.keras.autolog()
        mlflow.log_param('loss',configs['loss'])
        mlflow.log_param('epochs',configs['epochs'])
        mlflow.log_param('batch_size',configs['batch_size'])
        mlflow.log_param('data_path',configs['data_filepath'])

        # Instantiates Keras callbacks
        model_save_path = ".saved_models/model.h5"
        checkpoint = ModelCheckpoint(model_save_path,
                                monitor='val_loss',
                                mode='min',
                                save_best_only='True',
                                verbose=1)

        # Preprocesses data
        processor = DataProcessor(configs['data_filepath'])
        samples_train, samples_test, labels_train, labels_test, snr_train, snr_test = processor.process()

        # DEFINE YOUR MODEL HERE:
        model = Sequential()
        model.add(Dense(32, activation='relu',input_shape=(512,)))
        model.add(Dense(4, activation='softmax'))
        model.compile(loss=configs['loss'], optimizer=configs['optimizer'], metrics=['accuracy'])

        plot_path = '.model_plot.png'
        plot_model(model, to_file=plot_path, show_shapes=True, show_layer_names=True)

        # Trains model
        model.fit(samples_train, labels_train,
                  batch_size=configs['batch_size'],
                  epochs=configs['epochs'],
                  verbose=0,
                  validation_split=0.2,
                  callbacks=[checkpoint])

        mlflow.log_artifact(model_save_path)
        mlflow.log_artifact(plot_path)

        os.remove(model_save_path)
        os.remove(plot_path)

        # Tests model and scores
        user_input = input("Do you want test your model On the Test Set? (y/n) ")

        if user_input == 'y':
            score = model.evaluate(samples_test, labels_test, verbose=0)
            print('Test loss', score[0])
            print('Test accuracy', score[1])



if __name__ == '__main__':
    main()
