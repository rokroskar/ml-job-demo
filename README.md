# RenkuLab non-interactive PyTorch MNIST job demo

This project demonstrates how to run a non-interactive training job on RenkuLab.
It trains a small PyTorch neural network on the MNIST handwritten-digits dataset and writes the trained model, metrics, plots, and predictions to an output folder.

The project is intentionally small. The repository only contains:

- `requirements.txt` — the Python dependencies used to build the image
- `train.py` — the training script

## What this example shows

This example combines three RenkuLab features:

1. **Build from code** — RenkuLab builds a Python image from this repository.
2. **Data connectors** — the MNIST input data is mounted from Zenodo through a Renku DOI data connector.
3. **Non-interactive jobs** — the training script runs as a batch job instead of an interactive session.

The MNIST data comes from this Zenodo record:

<https://zenodo.org/records/10058130>

DOI: `10.5281/zenodo.10058130`

MNIST is a classic handwritten-digit image dataset. It is similar in spirit to the smaller optical-digits dataset often used in scikit-learn examples, but it contains larger 28×28 grayscale images.

## Data connector

The project links a global DOI data connector named **MNIST dataset**. In the job, RenkuLab mounts it at:

```text
/home/renku/work/mnist-dataset-doi-10.5281-zenodo.10058130
```

The training script looks for the standard MNIST files in that mounted folder, for example `train-images-idx3-ubyte.gz` and `train-labels-idx1-ubyte.gz`.

When running locally, that RenkuLab mount path will not exist. In that case, the script automatically streams the same MNIST files directly from Zenodo, so local testing still works.

## Job launcher setup

The job does not use a `Procfile`. Instead, the RenkuLab launcher is configured to run the built image with the Cloud Native Buildpacks launcher and pass the training command as arguments.

Conceptually, the launcher runs:

```bash
python /home/renku/work/ml-job-demo/train.py \
  --dataset mnist \
  --output-path /home/renku/work/output \
  --epochs 10
```

The image entrypoint is set to the CNB launcher:

```text
/cnb/lifecycle/launcher
```

This makes the job command explicit in the launcher configuration and avoids relying on Procfile process types.

## Outputs

The job writes the following artifacts to the configured output folder:

- `model.pt` — trained PyTorch model state and metadata
- `metrics.json` — accuracy and classification metrics
- `report.md` — short human-readable summary
- `training_history.csv` — loss and accuracy by epoch
- `training_curve.png` — training progress plot
- `confusion_matrix.png` — test-set confusion matrix
- `sample_predictions.png` — sample digit predictions
- `predictions.csv` — test-set predictions

The job logs also print the final accuracy and total runtime.

## Running the example on RenkuLab

1. Open the project: <https://renkulab.io/p/rok.roskar/jobs-example>
2. Copy the project to your own RenkuLab namespace.
3. Install the Renku CLI, `rnk`, from <https://github.com/swissdatasciencecenter/renku-cli>.
4. Log in to RenkuLab from your terminal:

   ```bash
   rnk login
   ```

5. In your copied project, open the job/session launcher and copy its launcher ID from the URL bar.
6. Start the non-interactive job:

   ```bash
   rnk job start <LAUNCHER_ID>
   ```

Useful commands:

```bash
rnk job list
rnk job logs <JOB_NAME>
```

When the job finishes, inspect the output folder to see the generated model, metrics, plots, and report.

## Rebuilding the image

The current launcher uses a previously built image as a custom image so that the command and arguments can be set explicitly.

If you change `requirements.txt` or otherwise need to rebuild the image:

1. Switch the launcher back to build-from-code.
2. Rebuild the image from this repository.
3. Switch the launcher back to a custom image using the newly built image.
4. Keep the command as `/cnb/lifecycle/launcher` and the arguments as the training command shown above.

## Current limitations

Non-interactive jobs are currently an alpha feature. Known limitations:

- logs may not persist for a predictable amount of time
- only run one job per launcher
- command-line arguments are configured in the launcher, not dynamically from `rnk`
- launcher configuration is currently done in the UI/API rather than the CLI
- UI support for non-interactive jobs is still evolving
- the built image includes the selected frontend even if the job does not use it

## Local test

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py --dataset mnist --output-path public --epochs 1
```

Locally, the script will stream MNIST from Zenodo if the RenkuLab connector mount is not available.
