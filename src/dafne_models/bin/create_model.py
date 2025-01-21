#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import os
import pickle
import sys
import time
import uuid

from dafne_dl.DynamicDLModel import source_to_fn, DynamicDLModel

assert sys.version_info.major == 3, "This software is only compatible with Python 3.x"

if sys.version_info.minor < 9:
    from dafne_models.utils.source_tools import extract_function_source_basic as extract_function_source
else:
    from dafne_models.utils.source_tools import extract_function_source

from dafne_models import resources

if sys.version_info.minor < 10:
    import importlib_resources as pkg_resources
else:
    import importlib.resources as pkg_resources

import matplotlib.pyplot as plt
import numpy as np
from dafne_dl.common.DataGenerators import DataGeneratorMem
from dafne_dl.common.preprocess_train import common_input_process_single, input_creation_mem, weighted_loss
from dafne_dl.labels.utils import invert_dict
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import Callback

FIXED_MODEL_SIZE = (256, 256)
BATCH_SIZE = 5
MAX_EPOCHS = 400
PATIENCE = 5
ENABLE_GUI = False
STORE_PREPROCESS = True
FORCE_PREPROCESS = False
PREPROCESS_ONLY = False
MIN_EPOCHS = 10
MAX_INPUT_SIZE = 256
DEFAULT_BIASCORRECTION_LEVELS = 4
DEFAULT_BIASCORRECTION_NORMALIZE = -1

DATA_PATH = None


def set_data_path(path):
    global DATA_PATH
    DATA_PATH = path


def get_data_path():
    return DATA_PATH


def set_force_preprocess(force):
    global FORCE_PREPROCESS
    FORCE_PREPROCESS = force


def get_force_preprocess():
    return FORCE_PREPROCESS


# Get the path of the script's parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# Add the parent directory to sys.path to import common
sys.path.append(parent_dir)


def load_data(data_path):
    """
    Load all the npz files in the folder
    :param data_path: string containing the path to the folder
    :return: list of numpy file objects
    """
    data_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npz')]
    data_list = []
    for file in data_files:
        print(f'Loading {file}')
        npz_file = np.load(file)
        data_list.append(npz_file)
    return data_list


def sanitize_label(label):
    if label.endswith('_L'):
        label = label[:-2] + '_X'
    elif label.endswith('_R'):
        label = label[:-2] + '_Y'
    return label


def get_model_info(data_list):
    """
    Get the common resolution, size, and labels from the data
    :param data_list: the list of numpy file objects
    :return: common_resolution, model_size, label_dict
    """
    resolutions = [data['resolution'][:2] for data in data_list]
    common_resolution = np.median(resolutions, axis=0)[:2]

    max_size = [np.ceil(data['data'].shape[:2] * data['resolution'][:2] / common_resolution).max() for data in
                data_list]
    max_size = int(max(max_size))
    if max_size > MAX_INPUT_SIZE:
        zoom_factor = MAX_INPUT_SIZE / max_size
        common_resolution = common_resolution / zoom_factor
        max_size = MAX_INPUT_SIZE

    model_size = [max_size, max_size]

    labels = set()
    for data in data_list:
        for key in data.keys():
            if key.startswith('mask_'):
                label = key[5:]
                labels.add(label)

    labels = sorted(labels)

    n_labels = len(labels)
    missing_data_list = []
    for i, data in enumerate(data_list):
        n_existing_labels = 0
        for label in labels:
            if f'mask_{label}' not in data.keys() or data[f'mask_{label}'].sum() < 5:
                print(f'Warning: dataset {i} is missing label {label}')
            else:
                n_existing_labels += 1
        if n_existing_labels <= n_labels / 2:
            missing_data_list.append(i)

    for i in missing_data_list[::-1]:
        data_list.pop(i)

    label_dict = {i + 1: sanitize_label(label) for i, label in enumerate(labels)}
    return common_resolution, model_size, label_dict


def escape_string(string):
    """
    Escape a string to be used in python source code
    :param string: the string to escape
    :return: the escaped string
    """
    return string.replace('"', '\\"')


def get_model_functions(source):
    """
    Extract the create_model function from the source code
    :param source: source in string form
    :return: functions returning a keras model, the apply function, and the incremental learn function
    """
    model_create_fn = source_to_fn(extract_function_source(source, 'make_unet'))
    model_apply_fn = source_to_fn(extract_function_source(source, 'model_apply'))
    model_incremental_mem_fn = source_to_fn(extract_function_source(source, 'model_incremental_mem'))
    return model_create_fn, model_apply_fn, model_incremental_mem_fn


def convert_3d_mask_to_slices(mask_dictionary):
    """
    Convert a dictionary of 3d masks to list of dictionaries of 2d masks
    :param mask_dictionary: dictionary in the form {label: mask_3d}
    :return: a list of dictionaries in the form {label: mask_2d}
    """
    mask_list = []
    for i in range(mask_dictionary[list(mask_dictionary.keys())[0]].shape[2]):
        mask_list.append(
            {sanitize_label(label): (mask[:, :, i] > 0).astype(np.uint8) for label, mask in mask_dictionary.items()})
    return mask_list


def normalize_training_data(data_list, common_resolution, model_size, label_dict):
    """
    Normalize the training data
    :param data_list: the list of numpy file objects
    :param common_resolution: the common resolution of the data
    :param model_size: the common size of the data
    :param label_dict: the dictionary with the labels
    :return: list of normalized data, list of normalized masks
    """
    inverse_label_dict = invert_dict(label_dict)
    all_slice_list = []
    all_masks_list = []

    for i, data in enumerate(data_list):
        print('Normalizing', i + 1, '/', len(data_list))
        img_3d = data['data']
        image_list = [img_3d[:, :, i].astype(np.float32) for i in range(img_3d.shape[2])]
        training_data_dict = {'image_list': image_list, 'resolution': data['resolution'][:2]}
        mask_dictionary = {key[5:]: data[key] for key in data.keys() if key.startswith('mask_')}
        mask_list = convert_3d_mask_to_slices(mask_dictionary)

        processed_image_list, processed_mask_list = common_input_process_single(inverse_label_dict,
                                                                                common_resolution,
                                                                                model_size,
                                                                                model_size,
                                                                                training_data_dict,
                                                                                mask_list,
                                                                                False,
                                                                                DEFAULT_BIASCORRECTION_LEVELS,
                                                                                DEFAULT_BIASCORRECTION_NORMALIZE)
        all_slice_list.extend(processed_image_list)
        all_masks_list.extend(processed_mask_list)

    return all_slice_list, all_masks_list


def generate_training_and_weights(data_list, mask_list, band=49):
    return input_creation_mem(data_list, mask_list, band=band)


def make_validation_list(data_list, common_resolution, model_size, label_dict):
    """
    Create a validation list
    :param data_list: the list of datasets
    :param common_resolution: resolution of the model
    :param model_size: size of the model
    :param label_dict: dictionary of the labels
    :return:
    """
    if STORE_PREPROCESS and DATA_PATH and os.path.exists(
            os.path.join(DATA_PATH, 'validation_obj.pickle')) and not FORCE_PREPROCESS:
        with open(os.path.join(DATA_PATH, 'validation_obj.pickle'), 'rb') as f:
            training_objects = pickle.load(f)
    else:
        normalized_data_list, normalized_mask_list = normalize_training_data(data_list,
                                                                             common_resolution,
                                                                             model_size, label_dict)
        if not normalized_data_list:
            print('Warning! No valid validation data found!')
            return [], []
        training_objects = generate_training_and_weights(normalized_data_list, normalized_mask_list)
        if STORE_PREPROCESS and DATA_PATH:
            with open(os.path.join(DATA_PATH, 'validation_obj.pickle'), 'wb') as f:
                pickle.dump(training_objects, f)
    x_list = [np.stack([training_object[:, :, 0], training_object[:, :, -1]], axis=-1) for training_object in
              training_objects]
    y_list = [training_object[:, :, 1:-1] for training_object in training_objects]
    return x_list, y_list


def make_data_generator(data_list, common_resolution, model_size, label_dict):
    """
    Create a data generator
    :param data_list: the list of datasets
    :param common_resolution: resolution of the model
    :param model_size: size of the model
    :param label_dict: dictionary of the labels
    :return:
    """
    if STORE_PREPROCESS and DATA_PATH and os.path.exists(
            os.path.join(DATA_PATH, 'training_obj.pickle')) and not FORCE_PREPROCESS:
        with open(os.path.join(DATA_PATH, 'training_obj.pickle'), 'rb') as f:
            training_objects = pickle.load(f)
    else:
        print("Normalizing data...")
        normalized_data_list, normalized_mask_list = normalize_training_data(data_list,
                                                                             common_resolution,
                                                                             model_size, label_dict)
        print("Generating training objects...")
        training_objects = generate_training_and_weights(normalized_data_list, normalized_mask_list)
        if STORE_PREPROCESS and DATA_PATH:
            with open(os.path.join(DATA_PATH, 'training_obj.pickle'), 'wb') as f:
                pickle.dump(training_objects, f)
    steps = int(len(training_objects) / BATCH_SIZE)
    data_generator = DataGeneratorMem(training_objects, list_X=list(range(steps * BATCH_SIZE)),
                                      batch_size=BATCH_SIZE, dim=model_size)

    return data_generator, steps


def prepare_data(training_data_list, validation_data_list, common_resolution, model_size, label_dict):
    training_generator, steps = make_data_generator(training_data_list, common_resolution, model_size, label_dict)

    if len(validation_data_list) > 0:
        x_val_list, y_val_list = make_validation_list(validation_data_list, common_resolution, model_size, label_dict)
    else:
        x_val_list = []
        y_val_list = []

    return training_generator, steps, x_val_list, y_val_list

def train_model(model, training_generator, steps, x_val_list, y_val_list, custom_callbacks=None, base_model=None,
                num_trainable_layers=None):
    """
    Train the model
    :param model: Keras model
    :param training_generator: Training data generator
    :param steps: Steps per epoch
    :param x_val_list: Validation inputs
    :param y_val_list: Validation labels
    :param custom_callbacks: List of callbacks
    :param base_model: Path to the base model for transfer learning
    :param num_trainable_layers: Number of trainable layers for fine-tuning
    :return: the trained model
    """
    n_validation = len(x_val_list)
    TRANSFER_LEARNING = base_model is not None

    if TRANSFER_LEARNING:
        # Load pre-trained CHAOS model
        with open(base_model, 'rb') as f:
            old_dafne_model = DynamicDLModel.Load(f)

        old_model = old_dafne_model.model

        print('Total layers in current model:', len(model.layers))
        print('Transferring weights from base model...')
        current_layer = 0
        for layer, base_model_layer in zip(model.layers[:-1], old_model.layers):
            print(f"Setting weights for layer {current_layer}")
            current_layer += 1
            try:
                layer.set_weights(base_model_layer.get_weights())
            except ValueError:
                print(f"Skipping incompatible layer: {layer.name}")
                break
            layer.trainable = False  # Initially freeze all transferred layers

        model.layers[-1].trainable = True  # Keep the last layer trainable by default

        # Fine-tuning at a specific point if num_trainable_layers is provided
        if num_trainable_layers is not None:
            total_layers = len(model.layers)  # Total layers in the model

            # Ensure the user-provided value is valid
            if num_trainable_layers < 0 or num_trainable_layers > total_layers:
                raise ValueError(
                    f"Invalid number of trainable layers: {num_trainable_layers}. It must be between 0 and {total_layers}."
                )

            # Calculate the starting index to unfreeze
            fine_tune_at = total_layers - num_trainable_layers
            print(f"Fine-tuning from layer index: {fine_tune_at}")

    ######################################

    # Compile the model
    adamlr = optimizers.Adam(learning_rate=0.009765, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=True)
    model.compile(loss=weighted_loss, optimizer=adamlr)

    # Callbacks for training
    if n_validation > 0:
        class PredictionCallback(Callback):
            def __init__(self):
                super().__init__()
                self.min_val_loss = np.inf
                self.n_val_loss_increases = 0
                self.best_weights = None

            def on_epoch_end(self, epoch, logs=None):
                if epoch < MIN_EPOCHS:
                    return
                if logs['val_loss'] < self.min_val_loss:
                    self.min_val_loss = logs['val_loss']
                    self.n_val_loss_increases = 0
                    self.best_weights = self.model.get_weights()
                else:
                    self.n_val_loss_increases += 1

                if self.n_val_loss_increases >= PATIENCE:
                    self.model.stop_training = True

        prediction_callback = PredictionCallback()

        if custom_callbacks is None:
            custom_callbacks = [prediction_callback]

        history = model.fit(
            training_generator,
            epochs=MAX_EPOCHS,
            steps_per_epoch=steps,
            validation_data=(np.stack(x_val_list, 0), np.stack(y_val_list, 0)),
            callbacks=custom_callbacks,
            verbose=1
        )

        if prediction_callback.best_weights is not None:
            model.set_weights(prediction_callback.best_weights)
    else:
        if custom_callbacks is None:
            custom_callbacks = []
        history = model.fit(training_generator, epochs=MAX_EPOCHS, steps_per_epoch=steps, verbose=1, callbacks=custom_callbacks)

    return model, history



def create_model_source(model_name, common_resolution, model_size, label_dict, levels=5, conv_layers=2, kernel_size=2):
    """
    Create the source code of the model
    :param common_resolution: resolution of the model
    :param model_size: size of the model
    :param label_dict: dictionary of the labels
    :param levels: number of levels
    :return: the source code
    """
    if getattr(sys, '_MEIPASS', None):
        with open(os.path.join(sys._MEIPASS, 'resources', 'generate_model.py.tmpl'), 'r') as f:
            source = f.read()
    else:
        with pkg_resources.files(resources).joinpath('generate_model.py.tmpl').open() as f:
            source = f.read()

    model_uuid = str(uuid.uuid4())

    source = source.replace('%%MODEL_NAME%%', f'"{escape_string(model_name)}"')
    source = source.replace('%%MODEL_RESOLUTION%%', f'[{common_resolution[0]:.2f}, {common_resolution[1]:.2f}]')
    source = source.replace('%%MODEL_SIZE%%', f'[{model_size[0]:d}, {model_size[1]:d}]')
    source = source.replace('%%MODEL_UID%%', f'"{model_uuid}"')
    source = source.replace('%%LABELS_DICT%%', str(label_dict))
    source = source.replace('%%N_LEVELS%%', str(int(levels)))
    source = source.replace('%%N_CONV_LAYERS%%', str(int(conv_layers)))
    source = source.replace('%%INITIAL_KERNEL_SIZE%%', str(int(kernel_size)))
    source = source.replace('%%BIASCORRECTION_LEVELS%%', str(int(DEFAULT_BIASCORRECTION_LEVELS)))
    source = source.replace('%%BIASCORRECTION_NORMALIZE%%', str(int(DEFAULT_BIASCORRECTION_NORMALIZE)))

    return source, uuid


def create_model(model_name, data_path, levels=5, conv_layers=2, kernel_size=2, test_create_model=False):
    global DATA_PATH

    if test_create_model:
        print("Testing model creation")
        common_resolution = [0.5, 0.5]
        model_size = [321, 321]
        label_dict = {1: 'LK', 2: 'RK'}
    else:
        data_list = load_data(data_path)
        DATA_PATH = data_path
        common_resolution, model_size, label_dict = get_model_info(data_list)

    source, model_id = create_model_source(model_name, common_resolution, model_size, label_dict, levels, conv_layers,
                                           kernel_size)

    with open(f'generate_{model_name}_model.py', 'w') as f:
        f.write(source)

    create_model_function, apply_model_function, incremental_learn_function = get_model_functions(source)
    model = create_model_function()

    if test_create_model:
        return None, None

    n_datasets = len(data_list)
    validation_split = 0.2 if n_datasets < 10 else 0.1
    n_validation = int(n_datasets * validation_split)

    if n_validation == 0:
        print("WARNING: No validation data will be used")

    validation_data_list = data_list[:n_validation]
    training_data_list = data_list[n_validation:]

    training_generator, steps, x_val_list, y_val_list = prepare_data(training_data_list, validation_data_list,
                                                                     common_resolution, model_size, label_dict)

    if PREPROCESS_ONLY:
        return None, None

    trained_model, history = train_model(model, training_generator, steps, x_val_list, y_val_list)

    model_object = DynamicDLModel(model_id,
                                  create_model_function,
                                  apply_model_function,
                                  incremental_learn_function=incremental_learn_function,
                                  weights=trained_model.get_weights(),
                                  timestamp_id=int(time.time()))

    return model_object, history


def save_weights(model, model_name):
    """
    Save the weights of the model
    :param model: the model
    :param model_name: the name of the model
    :return:
    """
    os.makedirs('weights', exist_ok=True)
    model_path = os.path.join('weights', f'weights_{model_name}.weights.h5')
    model.save_weights(model_path)
    print(f'Saved weights to {model_path}')


def main():
    global ENABLE_GUI, FORCE_PREPROCESS, PREPROCESS_ONLY, MIN_EPOCHS

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="Name of the model to create")
    parser.add_argument("data_path", help="Path to data folder (containing the '*.npz' files)")
    parser.add_argument("--no-gui", "-n", help="Disable the GUI", action="store_true")
    parser.add_argument("--levels", "-l", help="Number of levels of the model", type=int, default=5)
    parser.add_argument("--conv-layers", "-c", help="Number of convolutional layers", type=int, default=2)
    parser.add_argument("--kernel-size", "-k", help="Initial size of the kernel", type=int, default=2)
    parser.add_argument("--preprocess-only", "-p", help="Only preprocess the data", action="store_true")
    parser.add_argument("--force-preprocess", "-f", help="Force preprocessing of the data", action="store_true")
    parser.add_argument("--min-epochs", "-m", help="Minimum number of epochs", type=int, default=10)
    parser.add_argument("--test-model", "-t", help="Test the model creation", action="store_true")
    args = parser.parse_args()

    if args.no_gui:
        ENABLE_GUI = False

    if args.force_preprocess:
        FORCE_PREPROCESS = True

    if args.preprocess_only:
        PREPROCESS_ONLY = True

    MIN_EPOCHS = args.min_epochs

    dl_model, history = create_model(args.model_name, args.data_path, args.levels, args.conv_layers, args.kernel_size,
                                     args.test_model)

    if PREPROCESS_ONLY or args.test_model:
        return

    save_weights(dl_model.model, args.model_name)

    with open(f'{args.model_name}.model', 'wb') as f:
        dl_model.dump(f)

    if ENABLE_GUI:
        plt.plot(history.history['loss'])
        try:
            plt.plot(history.history['val_loss'])
            plt.legend(['train', 'validation'], loc='upper left')
        except KeyError:
            pass
        plt.show()


if __name__ == '__main__':
    main()
