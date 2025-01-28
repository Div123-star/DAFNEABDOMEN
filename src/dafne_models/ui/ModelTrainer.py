import ast
import os
import time
import re

from dafne_dl import DynamicDLModel

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QObject, QVariant
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QSpinBox, QLabel


import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter
from tensorflow.keras.callbacks import Callback

from .ModelTrainer_Ui import Ui_ModelTrainerUI
from ..bin.create_model import load_data, get_model_info, create_model_source, get_model_functions, train_model, \
    prepare_data, set_data_path, set_force_preprocess
from ..utils.ThreadHelpers import separate_thread_decorator

PATIENCE = 10
MIN_EPOCHS = 20


class PredictionUICallback(Callback, QObject):
    fit_signal = pyqtSignal(float, float, np.ndarray, np.ndarray)

    def __init__(self, test_image=None):
        Callback.__init__(self)
        QObject.__init__(self)
        self.min_val_loss = np.inf
        self.n_val_loss_increases = 0
        self.test_image = test_image
        self.do_stop = False
        self.best_weights = None
        self.auto_stop_training = False

    @pyqtSlot(np.ndarray)
    def set_test_image(self, image):
        self.test_image = image

    @pyqtSlot(bool)
    def set_auto_stop_training(self, auto_stop):
        self.auto_stop_training = auto_stop

    @pyqtSlot()
    def stop(self):
        self.do_stop = True
        print('Stopping training...')

    def on_epoch_end(self, epoch, logs=None):
        if self.do_stop:
            self.model.stop_training = True
            return

        if 'val_loss' in logs:
            val_loss = logs['val_loss']
        else:
            val_loss = np.inf

        loss = logs['loss']

        if self.auto_stop_training:
            if epoch >= MIN_EPOCHS and val_loss < self.min_val_loss:
                self.min_val_loss = val_loss
                self.n_val_loss_increases = 0
                self.best_weights = self.model.get_weights()
            elif val_loss > self.min_val_loss:
                self.n_val_loss_increases += 1

            if self.n_val_loss_increases >= PATIENCE:
                self.model.stop_training = True
        else:
            self.best_weights = None
            self.min_val_loss = np.inf

        if self.test_image is None:
            self.fit_signal.emit(loss, val_loss, np.zeros((10, 10)))
            return

        segmentation = self.model.predict(np.expand_dims(self.test_image, 0))
        label = np.argmax(np.squeeze(segmentation[0, :, :, :-1]), axis=2)
        self.fit_signal.emit(loss, val_loss, self.test_image[:, :, 0], label)


class ModelTrainer(QWidget, Ui_ModelTrainerUI):
    set_progress_signal = pyqtSignal(int, str)
    start_fitting_signal = pyqtSignal()
    end_fitting_signal = pyqtSignal()

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.setupUi(self)

        self.setWindowTitle('Dafne Model Trainer')
        self.advanced_widget.hide()
        self.fit_output_box.hide()
        self.adjustSize()
        self.pyplot_layout = QtWidgets.QVBoxLayout(self.fit_output_box)
        self.pyplot_layout.setObjectName("pyplot_layout")
        self.data_dir = None
        self.fig = plt.figure()
        self.fig.tight_layout()
        self.fig.set_tight_layout(True)
        self.canvas = FigureCanvas(self.fig)
        self.ax_left = self.fig.add_subplot(121)
        self.ax_left_twin = self.ax_left.twinx()
        self.ax_right = self.fig.add_subplot(122)
        self.ax_right.set_title('Current output')
        self.ax_right.axis('off')

        self.pyplot_layout.addWidget(self.canvas)
        self.bottom_widget = QWidget()
        bottom_layout = QtWidgets.QHBoxLayout(self.bottom_widget)

        bottom_layout.addWidget(QtWidgets.QLabel('Show validation slice:'))
        self.slice_select_slider = QtWidgets.QSlider(self.bottom_widget)
        self.slice_select_slider.setOrientation(QtCore.Qt.Horizontal)
        bottom_layout.addWidget(self.slice_select_slider)
        self.slice_select_slider.valueChanged.connect(self.val_slice_changed)
        self.slice_select_slider.setRange(0, 0)
        self.slice_select_slider.setEnabled(False)

        self.auto_stop_training_checkBox = QtWidgets.QCheckBox('Auto stop training', self.bottom_widget)
        self.auto_stop_training_checkBox.setChecked(True)
        self.auto_stop_training_checkBox.stateChanged.connect(self.auto_stop_training_changed)
        bottom_layout.addWidget(self.auto_stop_training_checkBox)

        self.pyplot_layout.addWidget(self.bottom_widget)
        self.advanced_button.clicked.connect(self.show_advanced)
        self.choose_Button.clicked.connect(self.choose_data)
        self.save_choose_Button.clicked.connect(self.choose_save_location)
        self.set_progress_signal.connect(self.set_progress)
        self.start_fitting_signal.connect(self.start_fitting_slot)
        self.end_fitting_signal.connect(self.stop_fitting_slot)
        self.fit_Button.clicked.connect(self.fit_clicked)
        self.preprocess_Button.clicked.connect(self.preprocess_clicked)
        self.force_preprocess_check.stateChanged.connect(self.decide_enable_fit)
        self.is_fitting = False
        self.fitting_ui_callback = None
        self.loss_list = []
        self.val_loss_list = []
        self.current_val_slice = 0
        self.val_image_list = []
        self.preprocessed_data_exist = False
        ################################################################

        self.pushButton.setEnabled(False)  # Initially disabled
        self.pushButton.clicked.connect(self.choose_base_model)
        self.BaseModel_Text.textChanged.connect(self.decide_enable_fit)
        self.BaseModel_Text.textChanged.connect(self.on_text_changed)
        self.transfer_learning_checkBox.clicked.connect(self.toggle_fit_button)

        # Connect toggled signal to enable/disable widgets
        self.transfer_learning_checkBox.toggled.connect(self.set_transfer_learning_enabled)

        # Initialize widgets state based on the checkbox's initial state
        self.set_transfer_learning_enabled(self.transfer_learning_checkBox.isChecked())

    def set_transfer_learning_enabled(self, enabled):
        """
        Enable or disable all widgets inside the QHBoxLayout in transfer_learning_checkBox.
        """
        for i in range(self.horizontalLayout_6.count()):
            widget = self.horizontalLayout_6.itemAt(i).widget()
            if widget:
                widget.setEnabled(enabled)


    def on_text_changed(self):
        """Enable or disable the Fit button based on text in base_model_text."""
        if self.BaseModel_Text.text().strip():
            self.fit_Button.setEnabled(False)  # Disable the button if text is filled
        else:
            self.fit_Button.setEnabled(True)  # Enable the button if text is empty

    def toggle_fit_button(self):
        """
        Disable fit_button when transfer_learning_checkBox is clicked.
        """
        if self.transfer_learning_checkBox.isChecked():
            self.fit_Button.setEnabled(False)
        else:
            self.fit_Button.setEnabled(True)

    #############

    @pyqtSlot()
    def choose_base_model(self):
        """Open a file dialog and display the selected file path in lineEdit."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Base Model",
            "",
            "Model Files (*.model *.h5 *.pt);;All Files (*)"
        )
        if file_path:
            self.BaseModel_Text.setText(file_path)



        # Check if force_preprocess_check was changed
        elif self.sender() == self.force_preprocess_check:
            self.transfer_learning_checkBox.blockSignals(True)  # Prevent recursion
            self.transfer_learning_checkBox.setChecked(not bool(state))  # Invert the state
            self.transfer_learning_checkBox.blockSignals(False)



    @pyqtSlot(str, str, result=object)
    def extract_parameter(self, source_code, parameter_name):
        """Extract a parameter's value from the source code based on its assignment pattern."""
        pattern = rf"{parameter_name}\s*=\s*([^\n]+)"

        match = re.search(pattern, source_code)

        if match:
            value = match.group(1)
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return value.strip('"').strip("'")

        raise ValueError(f"Parameter '{parameter_name}' not found in source code")

    # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *

    @pyqtSlot()
    def show_advanced(self):
        self.advanced_widget.show()
        self.advanced_button.hide()
        self.adjustSize()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.is_fitting:
            event.ignore()

    def decide_enable_fit(self):
        if self.preprocessed_data_exist and not self.force_preprocess_check.isChecked():
            self.fit_Button.setText('Fit')
        else:
            self.fit_Button.setText('Preprocess + fit')
        if self.location_Text.text() and (self.model_location_Text.text() or self.BaseModel_Text.text()):
            self.fit_Button.setEnabled(True)
            if not self.preprocessed_data_exist or self.force_preprocess_check.isChecked():
                self.preprocess_Button.setEnabled(True)
            else:
                self.preprocess_Button.setEnabled(False)
        else:
            self.fit_Button.setEnabled(False)
            self.preprocess_Button.setEnabled(False)
            self.set_progress(0, '')

    #########################################################################
    def decide_preprocess(self):
        if os.path.exists(os.path.join(self.data_dir, 'training_obj.pickle')):
            self.preprocessed_data_exist = True
            self.force_preprocess_check.setEnabled(True)
            self.force_preprocess_check.setChecked(False)
        else:
            self.preprocessed_data_exist = False
            self.force_preprocess_check.setEnabled(False)
            self.force_preprocess_check.setChecked(True)

    @pyqtSlot()
    def choose_data(self):
        def invalid():
            self.save_choose_Button.setEnabled(False)
            self.location_Text.setText("")
            self.decide_enable_fit()
            self.data_dir = None

        npz_file, _ = QFileDialog.getOpenFileName(self, "Choose data files", "", "Numpy files (*.npz);;All Files(*)")

        if not npz_file:
            invalid()
            return

        self.data_dir = os.path.dirname(npz_file)

        npz_files = [f for f in os.listdir(self.data_dir) if f.endswith('.npz')]
        if len(npz_files) > 0:
            self.save_choose_Button.setEnabled(True)
            self.location_Text.setText(self.data_dir)
            self.decide_preprocess()
            self.decide_enable_fit()
        else:
            invalid()
            # show a warning dialog
            QMessageBox.warning(self, "Warning", "No npz files found in the selected directory")

    @pyqtSlot()
    def choose_save_location(self):
        def invalid():
            self.save_choose_Button.setEnabled(False)
            self.model_location_Text.setText("")
            self.decide_enable_fit()

        # Open a save file dialog with a default extension
        fileName, _ = QFileDialog.getSaveFileName(
            self,
            "Save Model As",
            "",
            "Model Files (*.model);;All Files (*)"
        )

        if fileName:
            # Extract the directory and file name
            self.model_dir = os.path.dirname(fileName)
            self.model_name = os.path.basename(fileName)

            # Ensure the file name ends with '.model'
            if self.model_name.endswith('.model'):
                self.model_name = self.model_name[:-6]

            # Set the selected file path in BaseModel_Text
            self.model_location_Text.setText(fileName)
            self.decide_enable_fit()
        else:
            invalid()

    @pyqtSlot()
    @pyqtSlot(bool)
    @separate_thread_decorator
    def fit(self, preprocess_only=False):
        self.is_fitting = True
        self.start_fitting_signal.emit()
        self.set_progress_signal.emit(0, 'Loading data')
        data_list = load_data(self.data_dir)
        set_data_path(self.data_dir)

        self.set_progress_signal.emit(10, 'Getting model info')
        common_resolution, model_size, label_dict = get_model_info(data_list)
        ##############################################################
        # TODO: check if transfer  learning is enabled
        # if enabled:
        # load the model:
        # Initialize parameters based on transfer learning state
        # Check if transfer learning is enabled via the checkbox
        if not self.transfer_learning_checkBox.isChecked():
            base_model = None

            levels = self.levels_spin.value()
            conv_layers = self.convlayers_spin.value()
            kernel_size = self.kernsize_spin.value()
            num_trainable_layers = None  # No transfer learning, so no fine-tuning
        else:
            base_model = self.BaseModel_Text.text().strip()

            if base_model and os.path.exists(base_model):

                with open(base_model, 'rb') as f:
                    old_model = DynamicDLModel.Load(f)

                source_code = old_model.init_model_function.source
                apply_source_code = old_model.apply_model_function.source

                levels = self.extract_parameter(source_code, 'N_LEVELS')
                conv_layers = self.extract_parameter(source_code, 'N_CONV_LAYERS')
                kernel_size = self.extract_parameter(source_code, 'INITIAL_KERNEL_SIZE')
                common_resolution = self.extract_parameter(apply_source_code, 'MODEL_RESOLUTION')
                model_size = self.extract_parameter(source_code, 'MODEL_SIZE')
           #############
                # Get number of trainable layers from spin box
                num_trainable_layers = self.num_trainable_layers_spinBox.value()
                print(f"Number of trainable layers set to: {num_trainable_layers}")
            ##########################
            else:
                QMessageBox.critical(self, "Error", "Invalid base model path.")
                return

        #############################
        set_force_preprocess(self.force_preprocess_check.isChecked())

        self.set_progress_signal.emit(20, 'Creating model')
        source, model_uuid = create_model_source(self.model_name, common_resolution, model_size, label_dict, levels,
                                                 conv_layers, kernel_size)

        # write the new model generator script
        with open(os.path.join(self.model_dir, f'generate_{self.model_name}_model.py'), 'w') as f:
            f.write(source)

        create_model_function, apply_model_function, incremental_learn_function = get_model_functions(source)
        model = create_model_function()

        n_datasets = len(data_list)
        if n_datasets < 10:
            validation_split = 0.2
        else:
            validation_split = 0.1

        n_validation = int(n_datasets * validation_split)

        if n_validation == 0:
            print("WARNING: No validation data will be used")

        validation_data_list = data_list[:n_validation]
        training_data_list = data_list[n_validation:]

        self.set_progress_signal.emit(30, 'Preparing data')

        print('preparing data')

        training_generator, steps, x_val_list, y_val_list = prepare_data(training_data_list, validation_data_list,
                                                                         common_resolution, model_size, label_dict)

        print(f'{x_val_list[0].shape=}')

        if preprocess_only:
            self.set_progress_signal.emit(100, 'Done')
            self.end_fitting_signal.emit()
            self.is_fitting = False
            QMessageBox.information(self, "Information", "Preprocess done")
            return

        self.set_progress_signal.emit(50, 'Training model')

        self.fitting_ui_callback = PredictionUICallback()
        self.val_image_list = x_val_list
        self.fitting_ui_callback.fit_signal.connect(self.update_plot)
        self.fitting_ui_callback.set_auto_stop_training(self.auto_stop_training_changed)
        self.slice_select_slider.setMaximum(len(x_val_list) - 1)
        if len(x_val_list) > 0:
            self.fitting_ui_callback.set_test_image(x_val_list[0])
            self.slice_select_slider.setEnabled(True)
        else:
            self.slice_select_slider.setEnabled(False)

        trained_model, history = train_model(model, training_generator, steps, x_val_list, y_val_list,
                                             [self.fitting_ui_callback], base_model, num_trainable_layers)
        if self.fitting_ui_callback.best_weights is not None:
            trained_model.set_weights(self.fitting_ui_callback.best_weights)

        self.fitting_ui_callback.deleteLater()
        self.fitting_ui_callback = None

        self.set_progress_signal.emit(90, 'Saving model')

        model_object = DynamicDLModel(model_uuid,
                                      create_model_function,
                                      apply_model_function,
                                      incremental_learn_function=incremental_learn_function,
                                      weights=trained_model.get_weights(),
                                      timestamp_id=int(time.time())
                                      )

        with open(os.path.join(self.model_dir, f'{self.model_name}.model'), 'wb') as f:
            model_object.dump(f)

        # save weights
        os.makedirs(os.path.join(self.model_dir, 'weights'), exist_ok=True)
        trained_model.save_weights(os.path.join(self.model_dir, 'weights', f'weights_{self.model_name}.weights.h5'))
        self.set_progress_signal.emit(100, 'Done')
        self.end_fitting_signal.emit()
        self.is_fitting = False
        # open a message box to show the user the model was saved
        QMessageBox.information(None, "Information", f"Model saved successfully as {self.model_name}.model")

    @pyqtSlot(int)
    def auto_stop_training_changed(self, checked):
        if self.fitting_ui_callback is not None:
            self.fitting_ui_callback.set_auto_stop_training(checked > 0)

    @pyqtSlot()
    def val_slice_changed(self):
        if not self.fitting_ui_callback:
            return
        try:
            self.fitting_ui_callback.set_test_image(self.val_image_list[self.slice_select_slider.value()])
        except IndexError:
            print("Validation slice out of range")

    @pyqtSlot(float, float, np.ndarray, np.ndarray)
    def update_plot(self, loss, val_loss, image, label):
        # Define the number of segments (including the transparent color for zero)
        num_segments = 21

        # Create a list of colors with varying alpha values
        colors = [(0, 0, 0, 0)]  # Transparent color for zero
        for i in range(1, num_segments):
            # Generate distinguishable colors using HSV color space
            color = plt.cm.get_cmap('hsv', num_segments)(i)
            color = (color[0], color[1], color[2], 0.5)
            colors.append(color)

        # Create the colormap
        labels_colormap = ListedColormap(colors)

        self.loss_list.append(loss)
        self.val_loss_list.append(val_loss)

        y_axis_formatter = FuncFormatter(lambda y, _: '{:.1g}'.format(y))

        self.ax_left.clear()
        self.ax_left.plot(self.loss_list, color='#E66100', label='train')
        self.ax_left.set_ylabel('Training loss', color='#E66100')
        self.ax_left.yaxis.set_major_formatter(y_axis_formatter)
        self.ax_left_twin.clear()
        self.ax_left_twin.plot(self.val_loss_list, color='#5D3A9B', label='validation')
        self.ax_left_twin.set_ylabel('Validation loss', color='#5D3A9B')
        self.ax_left_twin.yaxis.set_label_position('right')
        self.ax_left_twin.yaxis.set_major_formatter(y_axis_formatter)
        self.ax_right.clear()
        self.ax_right.set_title('Current output')
        self.ax_right.imshow(image, cmap='gray')
        self.ax_right.imshow(label, cmap=labels_colormap)
        self.ax_right.axis('off')
        self.canvas.draw()

    @pyqtSlot(int, str)
    def set_progress(self, progress, message=''):
        self.progressBar.setValue(progress)
        self.progress_label.setText(message)

    @pyqtSlot()
    def start_fitting_slot(self):
        self.loss_list = []
        self.val_loss_list = []
        self.fit_Button.setText('Stop fitting')
        self.preprocess_Button.setEnabled(False)
        self.fit_output_box.show()
        self.adjustSize()
        self.choose_Button.setEnabled(False)
        self.save_choose_Button.setEnabled(False)
        self.advanced_widget.setEnabled(False)
        self.advanced_button.setEnabled(False)
        if self.val_image_list:
            self.slice_select_slider.setEnabled(True)

    @pyqtSlot()
    def stop_fitting_slot(self):
        self.fit_Button.setText('Fit model')
        self.slice_select_slider.setEnabled(False)
        self.advanced_widget.setEnabled(True)
        self.advanced_button.setEnabled(True)
        self.decide_preprocess()
        self.decide_enable_fit()

    @pyqtSlot()
    def preprocess_clicked(self):
        self.fit(True)

    @pyqtSlot()
    def fit_clicked(self):
        if self.is_fitting:
            # stop the fitting
            if self.fitting_ui_callback is not None:
                self.fitting_ui_callback.stop()
                self.fit_Button.setText('Stopping...')
        else:
            self.fit()

def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName('Dafne Model trainer')
    window = ModelTrainer()
    window.show()
    sys.exit(app.exec_())
