import os

# --- Core Directories ---
DATA_DIR = "opencv_data"
FACES_DIR = os.path.join(DATA_DIR, "faces")
MODELS_DIR = os.path.join(DATA_DIR, "models")

# --- File Paths ---
LABELS_PATH = os.path.join(DATA_DIR, "labels.pkl")
TRAINER_PATH = os.path.join(DATA_DIR, "trainer.yml")
ID_MAP_PATH = os.path.join(DATA_DIR, "id_map.pkl")

# --- DNN Face Detector Models ---
# Proto: https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/deploy.prototxt
# Model: https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
FACE_PROTO = os.path.join(MODELS_DIR, "deploy.prototxt")
FACE_MODEL = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000_fp16.caffemodel")

# --- Gender Detection Models ---
# Proto: https://github.com/spmallick/learnopencv/raw/master/AgeGender/gender_deploy.prototxt
# Model: https://github.com/spmallick/learnopencv/raw/master/AgeGender/gender_net.caffemodel
GENDER_PROTO = os.path.join(MODELS_DIR, "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(MODELS_DIR, "gender_net.caffemodel")
GENDER_LIST = ['Male', 'Female']

# --- Recognizer & Detector Configuration ---
RECOGNITION_CONFIDENCE = 85
TRAINING_IMG_SIZE = (100, 100)
DNN_CONFIDENCE = 0.5

# Ensure directories exist
os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

