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

## Running a RenkuLab job

This project demonstrates how to run non-interactive RenkuLab jobs.

1. Open the project: <https://renkulab.io/p/rok.roskar/jobs-example>
2. Copy the project to your own RenkuLab namespace.
3. Install the Renku CLI on your local machine, `rnk` (version `>=0.4.0`), by downloading the appropriate binary from <https://github.com/swissdatasciencecenter/renku-cli/releases>.
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

Helpful commands:

* To check the job: `rnk job list`
* To fetch the logs: `rnk job logs <JOB_NAME>`

When the job finishes, the generated model, metrics, plots, and report are written to the `public/` output folder.

### Image configuration

The linked [repository](https://github.com/rokroskar/ml-job-demo) includes a python
`requirements.txt` and a `Procfile` - these are picked up by the image build process to

* install dependencies (`requirements.txt`)
* configure the job command (`Procfile`)

You can use a similar setup in your own projects to specify how the job should
run and with which packages.

### Note on launchers

The launcher in the project was built from code but then converted to an "external"
environment in order to specify the entrypoint. If you need to rebuild the image,
you have to switch it back to "build from code", rebuild, and then switch back
to "external" to set the entrypoint.


## Current limitations

Non-interactive jobs are currently an alpha feature - we are making them available
but hidden in order to gather experience and feedback.

Current known limitations:

* logs will not persist for a predictable amount of time; if the node that the
  job ran on is removed because of auto-scaling, the logs are gone

* only run one job per launcher

* no dynamic modifications to command line arguments from `rnk` - they have to
  be made in the launcher itself

* no launcher configuration from the CLI; you must use the UI to configure the
  launcher

* no support in the UI - this is coming in June, so stay tuned!

* the built image will always include the frontend even if not using it; we
  recommend using ttyd as it is very minimal and can be useful for debugging

## Local test

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py --dataset uci-optdigits --output-path public --epochs 30
```
