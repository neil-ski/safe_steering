# Minimum Safe Steer

This is the code that goes along with the blog post here [neil.ski/#/steer](https://www.neil.ski/#/steer)

# Run all experiments

You'll have to export your hugging face token as an environment variable:
```
export HF_TOKEN="your_actual_token"
```

I use Google Cloud Storage to store my experimental data and act as a checkpoint to provide fault tolerance. You'll need to add your Google Cloud credentials into a file called `gcs_key.json`

I've used a Docker container so you don't have to deal with guessing what versions of what dependencies I used.

```
docker pull nflugovoy/safe-steer:latest
```

Then you can run each experiment individually:

Experiment 1:
```
docker run --rm -it \
  --gpus all \
  -e HF_TOKEN \
  -v $(pwd)/gcs_key.json:/workspace/safe_steer/gcs_key.json \
  safe-steer python -m experiment_1

```

Experiment 2:
```
docker run --rm -it \
  --gpus all \
  -e HF_TOKEN \
  -v $(pwd)/gcs_key.json:/workspace/safe_steer/gcs_key.json \
  safe-steer python -m experiment_2

```

Experiment 3:
```
docker run --rm -it \
  --gpus all \
  -e HF_TOKEN \
  -v $(pwd)/gcs_key.json:/workspace/safe_steer/gcs_key.json \
  safe-steer python -m experiment_3

```

Experiment 4:
```
docker run --rm -it \
  --gpus all \
  -e HF_TOKEN \
  -v $(pwd)/gcs_key.json:/workspace/safe_steer/gcs_key.json \
  safe-steer python -m experiment_4

```

Because we are building the docker image on the same machine we are running it on, I've included a `.dockerignore` file to not copy your sensitive info like your `gcs_key.json` file and any big irrelevant files like `.git`. Feel free to modify the `.dockerignore` file to suit your needs.

# Building locally

First build the container:
```
git clone https://github.com/neil-ski/safe_steer.git

cd safe_steer

# this is your google cloud credentials that we use for the GCS bucket
vim gcs_key.json

docker build -t safe-steer .

docker tag safe-steer nflugovoy/safe-steer:latest

docker push nflugovoy/safe-steer:latest
```

you'll have to change the docker hub repo name to your account instead of nflugovoy

# High level design

I adapted the code from [https://github.com/james-oldfield/tpc](https://github.com/james-oldfield/tpc) from the paper [Beyond Linear Probes](https://arxiv.org/pdf/2509.26238) to extract the activations in the file `extract_activations.py` and train the linear model on the activations in `train_linear_model.py`.

I've tried to take out anything that isn't relevant to my experiments and add type hints wherever possible.

The original code memory maps the results into files. I stuck with this approach but added some fault tolerance by flushing the files and then writing them to a GCS bucket. 

If the script crashes and you restart it, it will look for the latest checkpoint for the experiment you are running, read the latest index and restart the experiment from there. It will also download the files containing the numpy arrays and write them to local files and use mmap again. 

It also stores the state of the random number generators so the experiment should be reproducible even with a crash.

Note that the file names don't have different run numbers or dates in them so if you run the same experiment multiple times, it will reuse the file from the previous run. Also running multiple experiments at the same time in the same container probably will have a race condition for the same reason. 

# Reproducibility

I've seeded all of the random number generators I know of and my checkpoints include the RNG values so that it should be reproducible. 