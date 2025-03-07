from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QLineEdit, QLabel, QVBoxLayout, QWidget, QComboBox, QMessageBox, QInputDialog
from PySide6.QtGui import QPixmap, QImage, QPalette, QBrush, QFont, QGuiApplication
from PySide6.QtWidgets import QHBoxLayout, QApplication, QFrame, QDialog
from PySide6.QtCore import Qt, QSize, QEasingCurve, QPropertyAnimation, QRect
from PySide6 import QtGui
from VisualStimuli import run_visual_stimuli
from cortex_live import Cortex
import os
import shutil
import json
import threading
import cortex
import tkinter as tk
import subprocess
from queue import Queue

current_dir = os.getcwd()


with open('profiles.json', 'r') as file:
    profiles = json.load(file)
current_profile = profiles[-1] # Get the current profile (assumes the current profile is the last one in the list) 
                                # If the current profile is not the last profile in the list, then get the first profile in the list current_profile = profiles[0]
data_dir = os.path.join(current_dir, 'Profile', current_profile)

ipc_queue = Queue()
cortex = Cortex(ipc_queue, data_dir)  # Create an instance of the Cortex class
current_dir = os.getcwd()

class RelaxWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Relax")
        self.data_thread = None # Initialize data_thread to None in __init__

        self.setWindowTitle("Relax")

        layout = QVBoxLayout()

        # Title
        title = QLabel('Relax')
        font = QFont("Arial", 20, QFont.Bold)
        title.setFont(font)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Subtitle
        subtitle = QLabel('When ready, please click on Next button')
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)

        # Next button
        next_button = QPushButton('Next')
        next_button.clicked.connect(self.on_next)
        layout.addWidget(next_button)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def on_next(self):
        start_event = threading.Event()  # Create an Event
        self.close()
        
        # Create an instance of Queue for inter-process communication
        ipc_queue = Queue()
        
        self.data_thread = threading.Thread(target=cortex.run, args=(ipc_queue,))
        self.data_thread.start()
    
       # Start data recording when the "Next" button is clicked
        self.data_thread = threading.Thread(target=run_visual_stimuli, args=(start_event,cortex, ipc_queue))
        self.data_thread.start()
        
        self.close()
        #run_visual_stimuli()
        
    def closeEvent(self, event):
        if self.data_thread and self.data_thread.is_alive():
            cortex.stop()  # Call the stop function in the cortex module to close the WebSocket connection
            self.data_thread.join()  # Wait for the data thread to finish
            
    
class SSVEPFinalWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Relax")

        layout = QVBoxLayout()

        # Title
        title = QLabel('Relax')
        font = QFont("Arial", 20, QFont.Bold)
        title.setFont(font)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Subtitle
        subtitle = QLabel('When ready, please click on Next button')
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)

        # Next button
        next_button = QPushButton('Next')
        next_button.clicked.connect(self.run_script)
        layout.addWidget(next_button)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def run_script(self):
        self.close()
        subprocess.Popen(['python', 'SSVEP_final.py'])
        
        
class ProfileSelectedWindow(QMainWindow):
    def __init__(self, profile, parent=None):
        super(ProfileSelectedWindow, self).__init__(parent)
        self.profile = profile
        self.data_dir = os.path.join(current_dir, "Profile", profile)  # Add this line to initialize data_dir
        print(f"Current data_dir: {self.data_dir}")
        
        # Write the selected profile and data directory to a JSON file
        config = {'current_profile': self.profile, 'data_dir': self.data_dir}
        with open('config.json', 'w') as f:
            json.dump(config, f)
        self.cortex_thread = None
        self.setWindowTitle("Welcome to your profile")

        layout = QVBoxLayout()

        # Title
        title = QLabel('Welcome to your profile')
        font = QFont("Arial", 20, QFont.Bold)
        title.setFont(font)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Mode of operation
        operation_label = QLabel('Please select the mode of operation')
        layout.addWidget(operation_label)

        # Operation Mode
        self.operation_mode_combo = QComboBox()
        self.operation_mode_combo.addItems(['Data Collection', 'GoPiGo Control', 'Classifier Training'])
        operation_mode_button = QPushButton('Enter')
        operation_mode_button.clicked.connect(self.select_operation_mode)
        layout.addWidget(self.operation_mode_combo)
        layout.addWidget(operation_mode_button)

        # Go Back Button
        go_back_button = QPushButton('Go Back')
        go_back_button.clicked.connect(self.close)
        layout.addWidget(go_back_button)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        
        # Create a separate thread for data recording
        self.data_thread = None
        
    def start_recording(self):
        self.queue = Queue()
        self.cortex = Cortex(self.queue, self.data_dir, "SSVEP_final")  # Pass the data_dir to Cortex
        self.cortex.start("Label")

    def select_operation_mode(self):
        operation_mode = self.operation_mode_combo.currentText()
        if operation_mode:
            if operation_mode == 'Data Collection':
                self.data_collection_window = DataCollectionWindow(parent=self)
                self.data_collection_window.show()
                
            elif operation_mode == 'GoPiGo Control':
                self.live_control_window = LiveControlWindow(parent=self)
                self.live_control_window.show()
                
            elif operation_mode == 'Classifier Training':
                self.offline_mode_window = OfflineModeWindow(parent=self)
                self.offline_mode_window.show()
                
            #QMessageBox.information(self, 'Success', f'Selected operation mode: {operation_mode}')
        else:
            QMessageBox.warning(self, 'Warning', 'Please select an operation mode')
            
    def closeEvent(self, event):
        if self.data_thread and self.data_thread.is_alive():
            cortex.stop()  # Call the stop function in the cortex module to close the WebSocket connection
            self.data_thread.join()  # Wait for the data thread to finish
            

class DataCollectionWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Data Collection")

        layout = QVBoxLayout()

        # Title
        title = QLabel('Welcome to Data Collection')
        font = QFont("Arial", 20, QFont.Bold)
        title.setFont(font)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Model selection
        self.model_combo = QComboBox()
        self.model_combo.addItem("Please select the Models")
        self.model_combo.addItems(['SSVEP', 'SSVEP_final'])  
        layout.addWidget(self.model_combo)

        # Select Model Button
        select_model_button = QPushButton('Enter')
        select_model_button.clicked.connect(self.select_model)
        layout.addWidget(select_model_button)

        # Go Back Button
        go_back_button = QPushButton('Go Back')
        go_back_button.clicked.connect(self.close)
        layout.addWidget(go_back_button)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def select_model(self):
        selected_model = self.model_combo.currentText()
        if selected_model and selected_model != "Please select the Models":
            if selected_model == 'SSVEP':
                self.relax_window = RelaxWindow(parent=self)
                self.relax_window.show()

                # Start data recording when SSVEP is selected
                #self.data_thread = threading.Thread(target=cortex.run, args=())
                #self.data_thread.start()
                
            elif selected_model == 'SSVEP_final':  # Added this elif block
                self.ssvep_final_window = SSVEPFinalWindow(parent=self)
                self.ssvep_final_window.show()
                
            elif selected_model == 'MI':
                mi_process = subprocess.Popen(['python', 'MI.py'])
                mi_process.wait()  # Wait for the MI process to finish
            
            elif selected_model == 'Hybrid':
                # Add code for the Hybrid option
                pass
            #QMessageBox.information(self, 'Success', f'Selected training mode: {selected_model}')
        else:
            QMessageBox.warning(self, 'Warning', 'Please select a model')
            
class GoPiGoControlWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("GoPiGo Control")
        self.process = None   # For holding the running process

        layout = QVBoxLayout()

        # Start Stimuli Button
        start_stimuli_button = QPushButton('Start Stimuli')
        start_stimuli_button.clicked.connect(self.start_stimuli)
        layout.addWidget(start_stimuli_button)

        # Start Cortex Button
        start_cortex_button = QPushButton('Start Cortex')
        start_cortex_button.clicked.connect(self.start_cortex)
        layout.addWidget(start_cortex_button)

        # Stop Cortex Button
        stop_cortex_button = QPushButton('Stop Cortex')
        stop_cortex_button.clicked.connect(self.stop_cortex)
        layout.addWidget(stop_cortex_button)

        self.setLayout(layout)

    def start_stimuli(self):
        subprocess.run(["python", "Test_live_visual.py"])

    def start_cortex(self):
        self.process = subprocess.Popen(["python", "cortex_live.py"])

    def stop_cortex(self):
        if self.process:
            self.process.terminate()
            
class LiveControlWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("GoPiGo Control")

        layout = QVBoxLayout()

        # Title
        title = QLabel('Welcome to Robot Control')
        font = QFont("Arial", 20, QFont.Bold)
        title.setFont(font)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Control GoPiGo Button
        control_gopigo_button = QPushButton('Control GoPiGo')
        control_gopigo_button.clicked.connect(self.control_gopigo)
        layout.addWidget(control_gopigo_button)

        # Go Back Button
        go_back_button = QPushButton('Go Back')
        go_back_button.clicked.connect(self.close)
        layout.addWidget(go_back_button)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def control_gopigo(self):
        self.gopigo_window = GoPiGoControlWindow(self)
        self.gopigo_window.show()


class OfflineModeWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Classifier Training")

        layout = QVBoxLayout()

        # Title
        title = QLabel('Welcome, train your classifiers')
        font = QFont("Arial", 20, QFont.Bold)
        title.setFont(font)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Classifier selection
        self.classifier_combo = QComboBox()
        self.classifier_combo.addItem("Please select the classifier")
        self.classifier_combo.addItems(['DWT_SVM','DWT_KNN','PSD_SVM','EEGNET','ChronoNet','RNN_GRU', 'RNN_LSTM'])
        layout.addWidget(self.classifier_combo)

        # Train Model Button
        train_model_button = QPushButton('Enter')
        train_model_button.clicked.connect(self.train_model)
        layout.addWidget(train_model_button)

        # Go Back Button
        go_back_button = QPushButton('Go Back')
        go_back_button.clicked.connect(self.close)
        layout.addWidget(go_back_button)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def train_model(self):
        selected_classifier = self.classifier_combo.currentText()
        if selected_classifier != "Please select the classifier":
            if selected_classifier == "CSP_SVM":  
                import subprocess
                subprocess.run(["python", "CSP_SVM.py"])
                QMessageBox.information(self, 'Model Training', 'CSP_SVM model trained. See command terminal for output.')

            elif selected_classifier == "FFT_SVM":  
                import subprocess
                subprocess.run(["python", "FFT_SVM.py"])
                QMessageBox.information(self, 'Model Training', 'FFT_SVM model trained. See command terminal for output.')

            elif selected_classifier == "CNN_EEGNET":
                import subprocess
                subprocess.run(["python", "CNN_EEGNET.py"])
                QMessageBox.information(self, 'Model Training',
                                        'CNN_EEGNET model trained. See command terminal for output.')
                
            elif selected_classifier == "EEGNET_B":
                import subprocess
                subprocess.run(["python", "data_import_eegnet.py"])
                QMessageBox.information(self, 'Model Training',
                                        'EEGNET_B model trained. See command terminal for output.')
                
            elif selected_classifier == "ChronoNet":
                import subprocess
                subprocess.run(["python", "data_import_chrononet.py"])
                QMessageBox.information(self, 'Model Training',
                                        'ChronoNet model trained. See command terminal for output.')
                
            elif selected_classifier == "PSD_SVM":
                import subprocess
                subprocess.run(["python", "PSD_SVM.py"])
                QMessageBox.information(self, 'Model Training',
                                        'PSD_SVM model trained. See command terminal for output.')
                
            elif selected_classifier == "RNN_GRU":  
                import subprocess
                subprocess.run(["python", "RNN_GRU.py"])
                QMessageBox.information(self, 'Model Training', 'RNN_GRU model trained. See command terminal for output.')
            elif selected_classifier == "RNN_LSTM":  
                import subprocess
                subprocess.run(["python", "RNN_LSTM.py"])
                QMessageBox.information(self, 'Model Training', 'RNN_LSTM model trained. See command terminal for output.')
            elif selected_classifier == "DWT_SVM":
                import subprocess
                subprocess.run(["python", "DWT_SVM.py"])
                QMessageBox.information(self, 'Model Training', 'DWT_SVM model trained. See command terminal for output.')

            elif selected_classifier == "LDA_SVM":  
                import subprocess
                subprocess.run(["python", "LDA_SVM.py"])
                QMessageBox.information(self, 'Model Training', 'LDA_SVM model trained. See command terminal for output.')
            elif selected_classifier == "FFT_LDA":  
                import subprocess
                subprocess.run(["python", "FFT_LDA.py"])
                QMessageBox.information(self, 'Model Training', 'FFT_LDA model trained. See command terminal for output.')
            elif selected_classifier == "FFT_KNN":  
                import subprocess
                subprocess.run(["python", "FFT_KNN.py"])
                QMessageBox.information(self, 'Model Training', 'FFT_KNN model trained. See command terminal for output.')

            elif selected_classifier == "DWT_KNN":
                import subprocess
                subprocess.run(["python", "DWT_KNN.py"])
                QMessageBox.information(self, 'Model Training', 'DWT_KNN model trained. See command terminal for output.')
            #else:
                #QMessageBox.information(self, 'Success', f'Selected classifier: {selected_classifier}')
        else:
            QMessageBox.warning(self, 'Warning', 'Please select a classifier')

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle('Cortex Application')  # give your window a title

        # Set window size.
        self.setFixedSize(1000, 800)
        
        # Set window icon
        icon_path = os.path.join(current_dir, "Visual", "sst-logo.png")
        self.setWindowIcon(QtGui.QIcon(icon_path))

        # Set the window background
        image_path = os.path.join(current_dir, "Visual", "UPB.png")
        oImage = QImage(image_path)
        #oImage = QImage("C:\\Users\\arkos\\Documents\\Project_Python_Code\\rs_repo\\Visual\\UPB.png")
        sImage = oImage.scaled(QSize(1000, 800))  # resize Image to widgets size
        palette = QPalette()
        palette.setBrush(QPalette.Window, QBrush(sImage))
        self.setPalette(palette)

        # Create a vertical layout
        layout = QVBoxLayout()

        # Create a label for the name
        self.label = QLabel('')
        layout.addWidget(self.label)
        
        self.select_profile_frame = QFrame()  # Create a QFrame
        self.select_profile_frame.setStyleSheet("background-color: white;")  # Set the background color to white
        self.select_profile_frame.setFixedHeight(30)
        self.select_profile_label = QLabel('Please select a profile')  # Create a QLabel with the prompt text
        layout.addWidget(self.select_profile_frame)  # Add the QFrame to the layout
        self.select_profile_frame.layout = QVBoxLayout()  # Create a QVBoxLayout for the QFrame
        self.select_profile_frame.layout.addWidget(self.select_profile_label)  # Add the QLabel to the QVBoxLayout
        self.select_profile_frame.setLayout(self.select_profile_frame.layout)  # Set the QVBoxLayout as the layout for the QFrame


        #self.select_profile_combo = QComboBox()
        #self.select_profile_combo.addItem('Please select a profile')  # Add the prompt as the first item
        #layout.addWidget(self.select_profile_combo)

        self.select_profile_combo = QComboBox()
        layout.addWidget(self.select_profile_combo)

        select_profile_button = QPushButton('Enter')
        select_profile_button.clicked.connect(self.select_profile)
        layout.addWidget(select_profile_button)

        create_profile_button = QPushButton('Create Profile')
        create_profile_button.clicked.connect(self.create_profile)
        layout.addWidget(create_profile_button)

        self.delete_profile_combo = QComboBox()  # Initialize delete_profile_combo
        layout.addWidget(self.delete_profile_combo)

        delete_profile_button = QPushButton('Delete Profile')
        delete_profile_button.clicked.connect(self.delete_profile)
        layout.addWidget(delete_profile_button)

        # Set the layout
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.profiles_path = "profiles.json"  # File to store profiles
        self.load_profiles()  # Load profiles when initializing the application

    def start_cortex_thread(self):
        profile_name = self.select_profile_combo.currentText()
        profile_dir = os.path.join(current_dir, "Profile", profile_name)
        if profile_name:
            self.profile_selected_window = ProfileSelectedWindow(profile_name, self.parent())
            self.profile_selected_window.show()
            self.hide()
            self.profile_selected_window.cortex_thread = Cortex(ipc_queue, "SSVEP_final", profile_dir)  # Added the profile_dir parameter
            self.profile_selected_window.cortex_thread.start("Label")

    def select_profile(self):
        self.profile = self.select_profile_combo.currentText()  # Store current profile in self.profile
        if self.profile:
            self.profile_selected_window = ProfileSelectedWindow(parent=self, profile=self.profile)  # Pass self.profile
            self.showMinimized()
            #self.hide()
            self.profile_selected_window.show()
        else:
            QMessageBox.warning(self, 'Warning', 'Please select a profile')
            
            
    def save_profiles(self):
        profiles = [self.select_profile_combo.itemText(i) for i in range(self.select_profile_combo.count())]
        with open(self.profiles_path, 'w') as file:
            json.dump(profiles, file)

    def load_profiles(self):
        profiles_dir = os.path.join(current_dir, "Profile")
        if os.path.exists(profiles_dir):
            profiles = [name for name in os.listdir(profiles_dir) if os.path.isdir(os.path.join(profiles_dir, name))]
            for profile in profiles:
                self.select_profile_combo.addItem(profile)
                self.delete_profile_combo.addItem(profile)
            with open(self.profiles_path, 'w') as file:
                json.dump(profiles, file)

    def create_profile(self):
        text, ok = QInputDialog.getText(self, 'Create Profile', 'Enter profile name:')
        if ok and text:
            profile_folder = os.path.join(current_dir, "Profile", text)
            os.makedirs(profile_folder)  # Create a new folder for the profile
            self.select_profile_combo.addItem(text)
            self.delete_profile_combo.addItem(text)
            self.save_profiles()  # Save profiles whenever a new profile is created

    def delete_profile(self):
        selected_profile = self.delete_profile_combo.currentText()
        if selected_profile:
            index = self.select_profile_combo.findText(selected_profile)
            if index >= 0:
                self.select_profile_combo.removeItem(index)
                self.delete_profile_combo.removeItem(index)
                profile_folder = os.path.join(current_dir, "Profile", selected_profile)
                shutil.rmtree(profile_folder)  # Delete the folder for the selected profile
            self.save_profiles()  # Save profiles whenever a profile is deleted
            
def main():
    app = QApplication([])
    window = MainWindow()
    
    # Get screen size to center the window
    screen = QGuiApplication.primaryScreen().geometry()
    window.setGeometry(0, 0, screen.width() / 3, screen.height() / 3)
    window.move((screen.width() - window.width()) / 2, (screen.height() - window.height()) / 2)

    # Define final geometry
    final_geometry = window.geometry()

    # Modify the initial geometry for the zoom effect
    window.setGeometry(QRect(final_geometry.x() + final_geometry.width() / 2, final_geometry.y() + final_geometry.height() / 2, 0, 0))
    window.show()

    # Create an animation for the window's geometry, for the zoom effect
    animation_geometry = QPropertyAnimation(window, b'geometry')
    animation_geometry.setDuration(1000)  # 1000ms = 1s
    animation_geometry.setStartValue(window.geometry())  # start from initial geometry
    animation_geometry.setEndValue(final_geometry)  # end at final geometry
    animation_geometry.setEasingCurve(QEasingCurve.InOutQuad)

    # Start the animations
    animation_geometry.start()

    app.exec()
    # You should check if the profile_selected_window attribute exists and then if the cortex_thread is alive
    if hasattr(window, 'profile_selected_window') and window.profile_selected_window.cortex_thread is not None and window.profile_selected_window.cortex_thread.is_alive():
        window.profile_selected_window.cortex_thread.stop()  # Stop the thread to stop data recording


if __name__ == '__main__':
    main()
