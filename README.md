# RenkuLab non-interactive PyTorch training job demo

This repository is a minimal example for running a non-interactive RenkuLab job with build-from-code. It contains only:

- `requirements.txt` for the Python environment
- `Procfile` for the default process/command
- `train.py` for the training workflow

The default job trains a small PyTorch neural network on the public UCI Optical Recognition of Handwritten Digits dataset. This is the external source of the same family of data used by scikit-learn's built-in digits example.

## Default command

The `Procfile` contains:

```Procfile
job: python train.py --dataset uci-optdigits --output-path public --epochs 30
```

Attach or create a `public` folder in the RenkuLab project if you want the generated artifacts to be published.

The default public data URL is:

```text
https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip
```

The script itself defaults to the current directory, so it can also be run locally without arguments:

```bash
python train.py
```

## Outputs

The job writes:

- `model.pt` with the trained PyTorch model state and metadata
- `metrics.json` with accuracy and a full classification report
- `report.md` with a human-readable summary
- `training_history.csv`
- `training_curve.png`
- `confusion_matrix.png`
- `sample_predictions.png` when using the built-in `sklearn-digits` option
- `predictions.csv`

## Using data from a Renku data connector

To highlight RenkuLab data connectors, attach an S3 data connector or another connector to the job and point the script at a mounted CSV file:

```bash
python train.py \
  --data-path /path/to/mounted/connector/dataset.csv \
  --target-column target \
  --output-path public \
  --epochs 30
```

The CSV workflow uses numeric columns as features and the selected target column as the label. For S3, prefer mounting the bucket/prefix as a Renku data connector and passing the mounted file path to `--data-path`.

## Running this as a RenkuLab job

This project is meant to be copied and run as a non-interactive RenkuLab job.

1. Open the project: <https://renkulab.io/p/rok.roskar/jobs-example>
2. Copy the project to your own RenkuLab namespace.
3. Install the Renku CLI on your local machine, `rnk`, from <https://github.com/swissdatasciencecenter/renku-cli>.
4. Log in to RenkuLab from your terminal:

   ```bash
   rnk login
   ```

5. In your copied project, open the job/session launcher you want to use and copy its launcher ID from the URL bar.
6. Start the non-interactive job from your terminal:

   ```bash
   rnk job start <LAUNCHER_ID>
   ```

The launcher uses the command from the `Procfile`:

```bash
python train.py --dataset uci-optdigits --output-path public --epochs 30
```

When the job finishes, the generated model, metrics, plots, and report are written to the `public/` output folder.

## Local test

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py --dataset uci-optdigits --output-path public --epochs 30
```
