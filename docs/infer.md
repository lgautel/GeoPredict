# Environment Setup

## 1. Download Checkpoints and Assets

Download the required model weights, simulation assets, and related files from the [GeoPredict-Robocasa Hugging Face repository](https://huggingface.co/Jingjing0601/GeoPredict-Robocasa).

Save all downloaded files into the `ckpts/` folder under the project root. The expected file structure is:

```text
GeoPredict/
├── ckpts/
│   ├── GeoPredict_robocasa.pth
│   ├── paligemma_tokenizer.model
│   ├── robocasa_norm_stats.json
│   └── robocasa/
├── data_processing/
├── docs/
├── models/
├── tools/
├── utils/
├── requirements.txt
└── test_robocasa.sh
```

## 2. Create Conda Environment

Create and activate a conda environment:

```bash
conda create -n geopredict python=3.10 -y
conda activate geopredict
```

Install the required dependencies:

```bash
cd /path/to/GeoPredict
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For RoboCasa installation and simulator setup, please follow the [official RoboCasa repository](https://github.com/robocasa/robocasa).

## 3. Run Evaluation

After preparing the environment and checkpoints, run:

```bash
bash test_robocasa.sh
```
