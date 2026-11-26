# PlantVillage Leaf Disease Classification

A comparative image-classification project using **classical machine learning (ML)** and **deep learning (DL)** on the PlantVillage leaf-disease dataset.

The project is organized into two notebooks:
- `plantvillage_pipeline_colab_more_embedding (3).ipynb` — data preparation, image embeddings, classical ML experiments, model selection, Optuna tuning, final ML evaluation, and export of the shared image splits.
- `plantvillage_DL.ipynb` — deep-learning baselines, transfer learning, fine-tuning, model evaluation, Grad-CAM explainability, robustness checks, real-photo testing, and a Streamlit application.

## Project goals

1. Prepare a reliable image dataset and inspect its quality and class distribution.
2. Create a reproducible, stratified train/validation/test split.
3. Compare classical ML models trained on features extracted by pretrained vision backbones.
4. Train and fine-tune deep-learning models using the same split when the same exported dataset is supplied.
5. Compare the final ML and DL approaches using held-out test data.

## Repository / input files

| File | Purpose |
|---|---|
| `plantvillage_pipeline_colab_more_embedding (3).ipynb` | Classical ML and shared data-preparation pipeline |
| `plantvillage_DL.ipynb` | Deep-learning experiments and Streamlit deliverable |
| `requirements.txt` | Python package list for the ML notebook environment |
| `classical_ml_pipeline_my_part.png` | Classical ML workflow diagram |

## Dataset and class-count note

The classical-ML notebook's introduction describes a **15-class** PlantVillage dataset. The deep-learning notebook detects the class count from the supplied folders, and its saved example summary reports results for a **38-class** run.

These are not automatically interchangeable. Before claiming a direct ML-vs-DL comparison, make sure both notebooks use the **same underlying images, class folders, labels, and train/validation/test membership**. The DL notebook reads an already split directory; it does not create a new split.

## Classical ML workflow

1. **Load and clean images**
   - Build a table of image paths and labels.
   - Check image integrity, dimensions and colour modes.
   - Remove byte-identical duplicates and conflicting labels.
2. **EDA**
   - Inspect class distribution, image dimensions, brightness, contrast and sample images.
3. **Stratified split**
   - Create train/validation/test partitions using a 70/15/15 split.
   - Keep the split CSVs as the shared source of truth.
4. **Training-only imbalance handling**
   - Calculate balanced class/sample weights, and/or
   - Create augmented copies for under-represented training classes.
   - Do not resample validation or test data.
5. **Embedding extraction**
   - Extract feature vectors using ResNet50, EfficientNet-B0, DenseNet121, ConvNeXt-Tiny and ViT-B/16.
   - Extract features for the original splits and for the augmented training copies.
6. **Baseline experiments**
   - Evaluate 16 classical ML models on each embedding and imbalance strategy.
   - Rank experiments using validation macro-F1.
7. **Embedding and model selection**
   - Select the best embedding using validation results.
   - Tune the top candidate models with Optuna and stratified cross-validation on training data only.
8. **Final ML model**
   - Refit the selected model on train + validation.
   - Evaluate on the test set once.

### Important ordering clarification

The updated workflow diagram places **imbalance handling before the embedding stage**. In the notebook implementation, the original train/validation/test embeddings are extracted first, then augmented training images are generated and embedded in a dedicated step. Functionally, the ML experiments use the original training embeddings **plus the embeddings of oversampled/augmented training copies**; validation and test remain unchanged. The diagram expresses the intended data flow rather than claiming that every notebook cell executes in that literal order.

## Deep-learning workflow

1. Mount Google Drive and point `DRIVE_DATA_ROOT` to a directory containing:
   `train/<class_name>/`, `val/<class_name>/`, and `test/<class_name>/`.
2. Load the supplied split and check path overlap and optional exact-duplicate overlap.
3. Decode and resize images to 224 × 224, cache them locally, and build `tf.data` input pipelines.
4. Train the baseline models:
   - MLP
   - Custom CNN
   - MobileNetV2 with a frozen pretrained backbone
   - EfficientNetB0 with a frozen pretrained backbone
5. Fine-tune the top layers of the pretrained models.
6. Evaluate models, inspect errors and confusion patterns, and generate Grad-CAM visualizations.
7. Compare augmentation vs. no augmentation and test robustness to image perturbations.
8. Optionally evaluate real leaf photos to examine the domain gap.
9. Build and package the Streamlit application.

The DL notebook's sample summary reports EfficientNetB0 fine-tuning as its best example run. Treat those figures as **illustrative results from that saved run**, not guaranteed results for a different dataset or a fresh execution.

## Running on Google Colab

### Classical ML notebook

1. Upload the notebook to Google Colab.
2. Run the setup cells and confirm the PlantVillage dataset root is detected correctly.
3. Check the configuration paths and output directory before running the full pipeline.
4. Run cells in order. Cached features and result CSVs are reused when available.
5. Keep the same random seed and split CSVs for all experiments.

### Deep-learning notebook

1. Upload the notebook to Google Colab.
2. Select **Runtime → Change runtime type → GPU** (for example, T4).
3. Mount Google Drive.
4. Set `DRIVE_DATA_ROOT` to the folder containing the three split directories.
5. Run the notebook from top to bottom. Models and histories are saved under `WORK_DIR` on Drive.

## Main outputs

Depending on the notebook configuration and the cells executed, outputs include:

- `data/meta.csv`, `data/clean.csv`
- `data/train.csv`, `data/val.csv`, `data/test.csv`
- `data/augmented/` and `data/train_extra.csv`
- `features/<embedding>/X_{train,val,test}.npy`
- `results/baselines_*.csv`, `results/embedding_comparison.csv`
- `results/tuned_models_*.csv`
- `config/best_embedding.json`, `config/best_ml_params.json`
- `models/best_ml_model.joblib`
- DL `.keras` model checkpoints, training histories and evaluation reports
- `streamlit_app/` containing the app, model, class names and sample images

## Evaluation and leakage precautions

- Split file paths before model fitting and preserve the same split for ML and DL.
- Fit learned preprocessing steps, class weights and resampling logic using training data only.
- Do not oversample validation or test data.
- Use validation macro-F1 for model/embedding selection.
- Keep the test set untouched until final evaluation.
- For a fair ML-vs-DL comparison, evaluate both final models on the exact same test images and label order.

## Notes about `requirements.txt`

The supplied requirements file includes the core scientific Python and ML packages, plus image and notebook utilities. It currently contains `ploty`; the package name for Plotly is normally `plotly`. Review and correct that entry before installing if Plotly is required. The DL notebook separately requires TensorFlow/Keras and should use a TensorFlow version compatible with the saved `.keras` model.

## Limitations and possible next steps

- PlantVillage images are captured in controlled conditions; performance may drop on field photos because of the domain gap.
- Visually similar diseases can be confused, especially within the same crop.
- Add representative real-world leaf images, targeted augmentation, and confidence calibration to improve practical reliability.
- Record the exact dataset version, class count, split sizes, random seed, package versions and final test metrics with the submitted results.
