# RenkuLab Job demo

This project demonstrates how use Renku Jobs on RenkuLab. 
It trains a small PyTorch neural network on the MNIST handwritten-digits dataset and writes the trained model, metrics, plots, and predictions to an output folder.

The project is intentionally small. The repository only contains:

- `requirements.txt` — the Python dependencies used to build the image
- `train.py` — the training script

## What this example shows

This example combines three RenkuLab features:

1. **Build from code** — RenkuLab builds a Python image from this repository.
2. **Data connectors** — the MNIST input data is mounted from Zenodo through a Renku DOI data connector.
3. **Renku Jobs** — the training script runs as a job instead of an interactive session.

## Data connector

The MNIST data comes from this Zenodo record:

<https://zenodo.org/records/10058130>

DOI: `10.5281/zenodo.10058130`

This is mounted read-only at:

  ```text
  /home/renku/work/mnist-dataset-doi-10.5281-zenodo.10058130
  ```

The training script looks for the standard MNIST files in the mounted MNIST folder, for example `train-images-idx3-ubyte.gz` and `train-labels-idx1-ubyte.gz`.

When running locally, the RenkuLab mount path will not exist. In that case, the script automatically streams the same MNIST files directly from Zenodo, so local testing still works.

By default, job outputs are written to `/home/renku/work/output` inside the job workspace. If you want outputs to persist outside the job, add a writable data connector such as Polybox or SWITCHdrive and mount it at `output`.

## Job launcher setup

The job launcher is configured to run: 

```bash
python /home/renku/work/ml-job-demo/train.py \
  --dataset mnist \
  --output-path /home/renku/work/output \
  --epochs 10
```

The job container's entrypoint is set to `python /home/renku/work/ml-job-demo/train.py` so the job arguments (e.g. the number of epochs) can easily 
be changed when the job is started. 

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
3. Optionally add a writable data connector and set it to mount at `output`.
4. Start the job or launch the interactive session from the launchers. 

## Local test

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py --dataset mnist --output-path public --epochs 1
```

When run locally, the script will stream MNIST from Zenodo if the RenkuLab connector mount is not available.
