# ArtEmis Speaker Tools B
This repo contains following things related to [2]:
1. User Interfaces used in human studies for MTurk Experiments
2. Evaluation Tools 
3. Neural Speakers (nearest neighbor baseline, basic & grounded versions of M2 transformers)

## Data preparation
Please, prepare annotations and detection features files for the ArtEmis dataset to run the code:
1. Download [Detection-Features](https://drive.google.com/file/d/1PJyaiuPgPAH8uwkAUzezvli89E4EJFSZ/view?usp=sharing) and unzip it to some folder. Features are computed with the code provided by [1].
2. Download [ pickle file](https://drive.google.com/file/d/1gjzGK-D9bqxPjjvYdM51sJSm3Vzvh59G/view?usp=sharing) which contains [<image_name>, <image_id>], and put it in the same folder where you have extracted detection features.
3. Download ArtEmis dataset.
4. Download vocabulary files [1](https://drive.google.com/file/d/1Diy2WRzZrQfTo7j2GdgTiDrY37s98slq/view?usp=sharing), [2](https://drive.google.com/file/d/1tm8gPufGErFe787pH4VBcHSvWw360NOK/view?usp=sharing)

Some bounding box visualizations for art images: 
<p align="center">
<img src="images/art_bbox.jpeg" alt="BBox Features" width=“850”/>
</p>

## Environment Setup
Clone the repository and create the `artemis-m2` conda environment using the `environment.yml` file:
```
conda env create -f environment.yml
conda activate artemis-m2
```

Then download spacy data by executing the following command:
```
python -m spacy download en
```

## Training procedure
Run `python train.py` using the following arguments:

| Argument | Possible values |
|------|------|
| `--exp_name` | Experiment name|
| `--batch_size` | Batch size (default: 10) |
| `--workers` | Number of workers (default: 0) |
| `--m` | Number of memory vectors (default: 40) |
| `--head` | Number of heads (default: 8) |
| `--warmup` | Warmup value for learning rate scheduling (default: 10000) |
| `--resume_last` | If used, the training will be resumed from the last checkpoint. |
| `--resume_best` | If used, the training will be resumed from the best checkpoint. |
| `--features_path` | Path to detection features file |
| `--annotation_folder` | Path to folder with COCO annotations |
| `--use_emotion_labels` | If enabled, emotion labels will be used (default: "False")|
| `--logs_folder` | Path folder for tensorboard logs (default: "tensorboard_logs")|


To train grounded-version of the model, include additional parameter `--use_emotion_labels=1`.
```
python train.py --exp_name <exp_name> --batch_size 50 --m 40 --head 8 --warmup 10000 --features_path /path/to/features --annotation_folder /path/to/annotations/artemis.csv --workers 4 --logs_folder /path/to/logs/folder [--use_emotion_labels=1]
```

## Pretrained Models
Download our pretrained models and put them under `saved_models` folder:
* [Basic M2 model (trained without emotion labels)](https://drive.google.com/file/d/1bNgOyGfTHUnhbiRCTUkcasMotvgCtW6N/view?usp=sharing)
* [Grounded M2 model (trained with emotion labels)](https://drive.google.com/file/d/1Flm_Xl60dQoWq2D98ABYR8Ag6tYiD-dk/view?usp=sharing)
* [Emotion Encoder for Grounded M2 model](https://drive.google.com/file/d/1nV2H8dMcmb3d_njyXtkxppGRcBzzo9t-/view?usp=sharing)

Run `python test.py` using the following arguments:

| Argument | Possible values |
|------|------|
| `--batch_size` | Batch size (default: 10) |
| `--workers` | Number of workers (default: 0) |
| `--features_path` | Path to detection features file |
| `--annotation_folder` | Path to folder with COCO annotations |

```
python test.py --exp_name <exp_name> --features_path /path/to/features --annotation_folder /path/to/annotations/artemis.csv --workers 4 [--use_emotion_labels=1]
```
Some generations from the neural speakers:
<p align="center">
<img src="images/m2_outputs.jpeg" alt="M2 outputs" width="850"/>
</p>

#### References
[1] [Faster R-CNN with model pretrained on Visual Genome](https://github.com/shilrley6/Faster-R-CNN-with-model-pretrained-on-Visual-Genome)<br>
[2] [ArtEmis: Affective Language for Visual Art (Panos Achlioptas, Maks Ovsjanikov, Kilichbek Haydarov, Mohamed Elhoseiny, Leonidas Guibas)
](https://arxiv.org/abs/2101.07396)<br>
[3][Meshed Memory Transformer](https://github.com/aimagelab/meshed-memory-transformer).