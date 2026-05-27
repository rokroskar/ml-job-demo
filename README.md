# RenkuLab non-interactive PyTorch MNIST job demo

This repository is a minimal example for running a non-interactive RenkuLab job with a Python environment built from code.

It contains:

- `requirements.txt` for the Python environment
- `train.py` for the training workflow

The job trains a small PyTorch neural network on MNIST. The input data is provided through a Renku global DOI data connector for this Zenodo record:

```text
https://zenodo.org/records/10058130
DOI: 10.5281/zenodo.10058130
```

MNIST is analogous to the earlier handwritten-digits example, but larger: 28x28 grayscale digit images instead of the smaller 8x8 optical-digits dataset.

## Data connector

The project links a global DOI data connector:

- Connector: `MNIST dataset`
- Connector ID: `01KSMPWWWAA5W1NTHWXM4TQ5BC`
- Project link ID: `01KSMPX86FC61GR89V4BPXAQTJ`
- Mounted path in the job: `/home/renku/work/mnist-dataset-doi-10.5281-zenodo.10058130`

The training script expects the standard MNIST IDX files in that mounted directory, for example:

- `train-images-idx3-ubyte` or `train-images-idx3-ubyte.gz`
- `train-labels-idx1-ubyte` or `train-labels-idx1-ubyte.gz`
- `t10k-images-idx3-ubyte` or `t10k-images-idx3-ubyte.gz`
- `t10k-labels-idx1-ubyte` or `t10k-labels-idx1-ubyte.gz`

## How the Renku launcher is configured

The image is built from this repository using `requirements.txt`. The job command is **not** configured with a `Procfile`. Instead, after the image has been built, the Renku launcher uses the CNB launcher directly.

Use this launcher environment configuration pattern:

```json
{
  "environment_image_source": "image",
  "environment_kind": "CUSTOM",
  "container_image": "<built-image>",
  "working_directory": "/home/renku/work",
  "mount_directory": "/home/renku/work",
  "command": ["/cnb/lifecycle/launcher"],
  "args": [
    "python",
    "/home/renku/work/ml-job-demo/train.py",
    "--dataset",
    "mnist",
    "--mnist-path",
    "/home/renku/work/mnist-dataset-doi-10.5281-zenodo.10058130",
    "--mnist-source",
    "connector",
    "--output-path",
    "/home/renku/work/output",
    "--epochs",
    "10"
  ]
}
```

This avoids relying on Procfile process types and makes the actual job command explicit in the launcher.

## Outputs

The job writes the following artifacts to the configured output path:

- `model.pt`
- `metrics.json`
- `report.md`
- `training_history.csv`
- `training_curve.png`
- `confusion_matrix.png`
- `sample_predictions.png`
- `predictions.csv`

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

Helpful commands:

```bash
rnk job list
rnk job logs <JOB_NAME>
```

When the job finishes, the generated model, metrics, plots, and report are written to the output folder configured in the launcher.

## Note on rebuilding the image

The current launcher uses a built image as an external/custom image so that the command and args can be set explicitly. If you change `requirements.txt` and need to rebuild the image, switch the launcher back to build-from-code, rebuild, then switch it back to an external/custom image using the built image and the command/args configuration above.

## Current limitations

Non-interactive jobs are currently an alpha feature. Known limitations:

- logs may not persist for a predictable amount of time
- only run one job per launcher
- command-line arguments are configured in the launcher, not dynamically from `rnk`
- launcher configuration is currently done in the UI/API rather than the CLI
- UI support for non-interactive jobs is still evolving
- the built image includes the selected frontend even if the job does not use it

## Local test

The default `--mnist-path` is the RenkuLab data connector mount:

```text
/home/renku/work/mnist-dataset-doi-10.5281-zenodo.10058130
```

For local testing, that mount will usually not exist, so the script can stream the MNIST files directly from Zenodo when the connector mount is not present:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py --dataset mnist --output-path public --epochs 1
```

In RenkuLab, the launcher should use `--mnist-source connector` so the job fails clearly if the data connector is not mounted.
