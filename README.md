# Face Mask Detection

Real-time face mask detection using a fine-tuned **MobileNetV2** CNN model with OpenCV webcam integration.

## Results
| Metric | Score |
|--------|-------|
| Accuracy | ~98% |
| Precision | ~97% |
| Recall | ~98% |

## Tech Stack
- Python 3.9+
- TensorFlow / Keras
- OpenCV
- MobileNetV2 (transfer learning)

## Setup
```bash
pip install -r requirements.txt
python model.py        # Train the model
python predict.py      # Run real-time webcam detection
```

## Dataset
Uses the [Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset) from Kaggle (~7,000 images).

## Project Structure
```
face_mask_detection/
├── model.py        # Model architecture & training
├── predict.py      # Real-time webcam inference
├── requirements.txt
└── README.md
```
