# deepfakeDetection

A deepfake detection project that prepares video data, detects faces, trains a CNN model, and predicts whether video content is fake or real.

## Repository Structure

- `00-convert_video_to_image.py` — convert video frames to images.
- `01-crop_faces_with_mtcnn.py` — detect and crop faces from images using MTCNN.
- `02-prepare_fake_real_dataset.py` — prepare datasets for fake vs real classification.
- `03-train_cnn.py` — train the CNN model on prepared face datasets.
- `predict_video.py` — run predictions on a video using the trained model.
- `requirements.txt` — Python dependencies.
- `env/` — local Python virtual environment.
- `fake_videos/`, `real_videos/`, `prepared_dataset/`, `split_dataset/`, `img/`, `output/` — project data and outputs.
- `model_checkpoint/` — trained model checkpoint storage.

## Setup

1. Create and activate a Python virtual environment.

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Usage

1. Convert videos to images:

```powershell
python 00-convert_video_to_image.py
```

2. Crop faces from images:

```powershell
python 01-crop_faces_with_mtcnn.py
```

3. Prepare the fake/real dataset:

```powershell
python 02-prepare_fake_real_dataset.py
```

4. Train the CNN model:

```powershell
python 03-train_cnn.py
```

5. Predict on a new video:

```powershell
python predict_video.py
```

## Notes

- Do not commit secrets or API keys to this repository.
- Keep your `env/` local and avoid pushing it to GitHub.
