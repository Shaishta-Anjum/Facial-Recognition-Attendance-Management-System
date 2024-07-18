# Face Recognition Attendance Management System

## Overview
The Face Recognition Based Attendance System is a comprehensive solution designed to streamline the process of attendance tracking using facial recognition technology. This system leverages advanced machine learning algorithms and computer vision techniques to provide a reliable, efficient, and user-friendly way to manage attendance records.

## Features
- **User Interface Module**: A secure and intuitive interface accessible via desktop and mobile devices.
- **Face Detection Module**: Utilizes Haarcascade and OpenCV for accurate face detection.
- **Recognition and Verification Module**: Employs machine learning for precise recognition and verification.
- **Attendance Logging Module**: Efficiently logs attendance records in real-time.

## How It Works
1. **Face Detection**: Captures images from the camera and detects faces using Haarcascade.
2. **Face Recognition**: Recognizes the detected faces using a pre-trained KNN model.
3. **Attendance Logging**: Logs the recognized faces' attendance with timestamps in a CSV file.

## Setup and Installation

### Prerequisites
- Python 3.6+
- Flask
- OpenCV
- NumPy
- Pandas
- Scikit-Learn
- Joblib

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/face-recognition-attendance-system.git
    cd face-recognition-attendance-system
    ```

2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Flask application:
    ```bash
    python app.py
    ```

## Usage
1. **Home Page**: Displays the current attendance records and provides options to take attendance or add new users.
2. **Add User**: Register new users by capturing multiple images of their faces.
3. **Take Attendance**: Start the attendance process by detecting and recognizing faces in real-time.

## Code Structure
- **app.py**: Main Flask application file handling routing and core functionality.
- **templates/**: HTML templates for rendering the web pages.
- **static/**: Static files including trained models and face images.

## Key Functions
- `extract_faces(img)`: Extracts faces from the input image using Haarcascade.
- `identify_face(facearray)`: Recognizes the face using the pre-trained KNN model.
- `train_model()`: Trains the KNN model on the registered users' face images.
- `add_attendance(name)`: Logs attendance for the recognized face.

## Proposed Methodology
- **Flask**: For web framework and application routing.
- **OpenCV**: For image processing and face detection.
- **Scikit-Learn**: For training and utilizing the KNN model.
- **Joblib**: For model serialization and deserialization.
- **Pandas**: For managing attendance records.
- **Camera**: As the primary hardware for capturing images.

## Hardware Implementation
- A standard webcam for capturing facial images.
- A computer or server to run the application.

## Conclusion
The Face Recognition Based Attendance System offers a modern, efficient, and accurate way to manage attendance. By integrating advanced technologies, it ensures a seamless and reliable attendance tracking experience, making it an ideal solution for educational institutions, workplaces, and other organizations.
