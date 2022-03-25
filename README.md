# SE&R 2022 Challenge - SER Track

- [Introduction](#Introduction)
- [Dependencies](#Dependencies)
- [Datasets](#Datasets)
- [Training](#Training)
- [Pre-Trained Model](#Pre-Trained-Model)
- [Contact](#Contact)

# Introduction

Automatic Speech Recognition for spontaneous and prepared speech & Speech Emotion Recognition in Portuguese (SE&R 2022) Workshop is the first edition of a new series of shared-tasks for the Portuguese language and introduces two versions of a new dataset called CORAA ([Corpus of Annotated Audios](https://sites.google.com/viw/tarsila-c4ai) built in the TaRSila project, an effort of the Center for Artificial Intelligence ([C4AI](http://c4ai.inova.usp.br/pt/nlp2-pt)).

The Speech Emotion Recognition track aims to motivate research for SER in our community, mainly to discuss theoretical and practical aspects of SER, pre-processing and feature extraction, and machine learning models for Portuguese. In this task, participants must train their own models using acoustic audio features using the dataset provided called [CORAA SER version 1.0](https://github.com/rmarcacini/ser-coraa-pt-br/) composed of approximately 50 minutes of audio segments labeled in three classes: neutral, non-neutral female, and non-neutral male. While the neutral class represents audio segments with no well-defined emotional state, the non-neutral classes represent segments associated with one of the primary emotional states in the speaker's speech.

This repository presents the code used by our team that got first place on the SER track.

Three key strategies make up the solution:

- The use of a multilingual model (Wav2vec2.0 XLS-R)
- The use of a mixture of voice emotion recognition datasets from several languages
- Dataset normalization


# Dependencies

It is important to install the dependencies before launching the application.

Run the following command to install the required dependencies using pip:

```
sudo pip install -r requeriments
```

# Datasets

Three more speech emotion recognition datasets were used in addition to the CORAA SER dataset:

- [EMOVO Corpus](https://aclanthology.org/L14-1478/)
- [RAVDESS](https://smartlaboratory.org/ravdess/)
- [BAVED](https://github.com/40uf411/Basic-Arabic-Vocal-Emotions-Dataset)

Run the script ```prepare_datasets.sh``` to download all of the datasets used in the experiments, then run the script ```get_metadata.py``` to prepare the metadata used to train, validate and test the models.

# Training

The ```default.yaml``` file in the ```config``` directory can be used to set the essential configurations for training a model.

If you want to apply gain normalization during training/testing, you must first calculate the mean dbfs level, which you can accomplish by running the ```get_mean_dbfs.py``` script in the ```utils``` directory, and then inserting the result in the ```target_dbfs``` parameter.

# Pre-Trained Model

The model weights with the best results can be found on the [huggingface hub](https://huggingface.co/alefiury/wav2vec2-xls-r-300m-pt-br-spontaneous-speech-emotion-recognition) and can be easily fine-tuned in more data by this application.

# Author

- Alef Iury Siqueira Ferreira

# Contact

- e-mail: alef_iury_c.c@discente.ufg.br
