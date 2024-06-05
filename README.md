<br />
<div align="center">
  <a href="https://github.zhaw.ch/mict/chall_data">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/ZHAW_Logo.svg/771px-ZHAW_Logo.svg.png" alt="Logo" width="103" height="120">
  </a>

  <h3 align="center">ChaLL: End-to-End Speech Recognition for Language Learning</h3>

  <p align="center">
   This repository contains the experiments of both a semester work (VT2) and a subsequent paper, focusing on enhancing automatic speech recognition for language learners within the ChaLL project.
   <br />
  </p>

  <p align="center">
    <a href="#about-the-project-chall">About The Project ChaLL</a> •
    <a href="#getting-started">Getting Started</a> •
    <a href="#usage">Usage</a> •
    <a href="#citation">Citation</a>
  </p>

</div>


> <details>
>
> <summary><b>VT2</b></summary><br>
>
> [Automatic Speech Processing for Language Learners](#)
>
> **Abstract:** The ChaLL project aims for a voice-based chatbot that provides Swiss primary school children with interactive speaking opportunities and situational feedback in second language (L2) acquisition. Within the scope of the ChaLL projects, this work deals with the unique challenges involved in accurately transcribing spontaneous English speech from non-native children who are learning it as their L2. Additionally, this is complicated by the need to preserve learners’ errors in the final transcript for subsequent feedback. 
> 
> As a prerequisite for this research, a corpus was first compiled that aligns audio data (collected in school as part of the ChaLL project) with externally created transcripts, including error annotations. Further, the Word Annotation Error Rate (AWER) has been defined as a metric to measure error-preservation. The corpus was used to evaluate existing STT systems’ performance (WER), their ability in preserving errors (AWER) and ultimately as input to fine-tune a pre-trained facebook/wav2vec2-xls-r-300m model. 
> 
> This study has successfully demonstrated the impact of using targeted data for fine-tuning a pre-trained model. Remarkably, even using the small 300 parameters model resulted in a marginal reduction in WER and a reduction in both the AWER for erroneous and L1 words. All systems, the trained one included, notably struggle with the latter. Future studies could enhance results by employing the larger 1B model, using data augmentation techniques to generate additional data, and specifically targeting improvements for (Swiss-)German words.
> </details>

> <details>
>
> <summary><b>ACL_2024</b></summary><br>
>
> [Error-preserving Automatic Speech Recognition of Young English Learners’ Language](#)
>
> **Abstract:** One of the central skills that language learners need to practice is speaking the language. Currently, students in school do not get enough speaking opportunities and lack conversational practice. Recent advances in speech technology and natural language processing allow for the creation of novel tools to practice their speaking skills. In this work, we tackle the first component of such a pipeline, namely, the automated speech recognition module (ASR), which faces a number of challenges: first, state-of-the-art ASR models are often trained on adult read-aloud data by native speakers and do not transfer well to young language learners’ speech. Second, most ASR systems contain a powerful language model, which smooths out mistakes made by the speakers. To give corrective feedback, which is a crucial part of language learning, the ASR systems in our setting need to preserve the mistakes made by the language learners. In this work, we build an ASR system that satisfies these requirements: it works on spontaneous speech by young language learners and preserves their mistakes. For this, we collected a corpus containing around 85 hours of English audio spoken by Swiss learners from grades 4 to 6 on different language learning tasks, which we used to train an ASR model. Our experiments show that our model benefits from direct fine-tuning on children’s voices and has a much higher error preservation rate than other models
> </details>


## About the Project ChaLL

> ChaLL, a voice-based chatbot that provides language learners
with opportunities to practice speaking in both focused and unfocused task-based conversations and
receive feedback, free from the time constraints and pressures of the traditional classroom setting.

### ChaLL Data

As part of the ChaLL project, data for speech-to-text (STT) research and error recognition analysis was gathered:

1) **Audio Files**: We collected audio recordings from educational interventions across different grades. Each file is identified by the school's postal code, grade, and an audio ID. Recording quality varies due to environment, equipment, and speakers' accents.

2) **Transcripts**: Transcripts pair spoken words with text and include metadata on recordings, speakers, and segmented speech. Each segment details words with timestamps, capturing nuances like repetitions, language-specific spellings, and non-standard speech elements:
   - *`@g` Swiss-German Words*: "He plays on the Wiese@g"
   - *`@?` Best Guess*: Used for uncertain transcriptions, e.g.: "It’s yellow and have@? a nice face."
   - *`@!` Speech Errors*: Marks learner's speech errors, e.g.: "He play@! soccer."
   - *`@!` Missing Words*: Indicated by @! in the correct place, e.g.: "What @! you doing?"
   - *`-` Repetitions*: Marked with a dash, e.g.: "He’s - he's really tall."
   - *`--` Reformulations*: Marked with double dashes, e.g.: "He’s -- he doesn’t like school."
   - *`(...)`Long Pauses*: Denoted with ellipses "(…)"

3) **Metadata**: An Excel file documents metadata for each audio recording, detailing the intervention, participants, recording conditions, and tasks. This enhances understanding of the recording context but doesn't link speakers to transcripts due to matching complexity.

### Data Preparation and Availability

Data processing for this project has been outsourced and divided into two resources for easy access: 
a [data preparation repository](https://github.zhaw.ch/mict/chall_data) (🔒) and a [Huggingface dataset](https://huggingface.co/datasets/mict-zhaw/chall).

- **Data Preparation Repository**: 
The data preparation repository is relevant if you wish to generate your own data. Follow the instructions provided in the [repository](https://github.zhaw.ch/mict/chall_data) to prepare the data. Note that access to the original data, as described in the previous chapter, is required.
This process has been outsourced to ensure anonymization. The data is anonymized in this repository, and having direct access to it would endanger this anonymity.

- **Huggingface Dataset**:
To use the prepared data, refer to the Huggingface dataset.
For ethical reasons, the data is hosted on a separate server and must be downloaded manually.
Please follow the instructions on the [model card](https://huggingface.co/datasets/mict-zhaw/chall) to download and access this data.
When using the Huggingface dataset, you have two configuration options: use the default configuration to access the data as used in the paper, or use a custom configuration to define how the data is prepared according to your specific needs.


### Model `ChaLL-300M`

[`ChaLL-300M`](https://huggingface.co/mict-zhaw/chall_wav2vec2_xlsr_300m) is a fine-tuned ASR model from the Wav2Vec-XLSR-300M base model.
This model represents the best performing fold (Fold 5) from a k-fold cross-validation training process as described in the paper.
It addresses the unique challenges of transcribing the spontaneous speech of young English learners by preserving their grammatical, lexical, and pronunciation errors.



<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

### Local Installation

To install the necessary dependencies for this pipeline, run:

```
pip install -r requirements.txt
```

### Using Docker


#### Docker Build

```shell
# Note: Run this command from the root directory
docker build -f Dockerfile -t mict/chall_stt   . 2>&1 | tee create_docker_log.txt
```

#### Docker Run

```shell
nvidia-docker run --shm-size=32g -v /cluster/data/mict:/work_space_data   -v /cluster/home/mict:/workspace -it mict/chall_stt  bash;
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

Follow the instructions below and use the provided configuration to replicate the experiments.

### Configuration

Use the provided configuration to replicate the results from the paper.

<details>
<summary><b>ACL_2024</b></summary>

#### `config/train/config.json`

*todo in this branch kfold is not yet implemented with new version of data and pipeline*

```json
{
  "experiment_name": "chall_paper",
  "experiment_tag": "feb7_5fold",

  "train_corpora": [["mict-zhaw/chall", ["train"]]],
  "eval_corpora": [["mict-zhaw/chall", ["eval"]]],

  "alt_base_path": "/work_space_data/chall",
  "data_dir": "../chall_data/",

  "dataset_kwargs": {
    "name": "asr_acl"
  },

  "seed": 123,
  "device": "cuda",
  "wav2vec_base_model": "facebook/wav2vec2-xls-r-300m",
  "checkpoint": null,

  "n_steps": 120,
  "batch_size": 14,
  "eval_batch_size": 1,
  "gradient_accumulation_steps": 15,
  "gradient_checkpointing": false,
  "learning_rate": 3e-05,
  "label_smoothing_factor": 0.0,
  "fp16": true,
  "bf16": false,
  "optim": "adamw_bnb_8bit",
  "freeze_w2vm": false,

  "save_total_limit": 2,
  "validation_freq": 100,
  "save_step": 100,
  "logging_step": 20,

  "n_train_samples": -1,
  "n_valid_samples": -1
}
```

#### `create_hypothesis_config.json`

```json
{
  "alt_base_path": "./",

  "in_corpora": [["chall", "chall_wav2vec2_xls_r_300m"]],
  "out_corpora": [["chall_predictions_paper", "chall_wav2vec2_xls_r_300m"]],

  "experiment_name": "chall_paper",
  "experiment_tag": "create_hypotheses",

  "systems": [
    "./models/chall_paper/built/chall_wav2vec2_xls_r_300m_fold_4/"
  ],

  "data_kwargs": {
    "min_length": 0.5,
    "interventions": ["8032_6_B", "8404_5", "8400_4"]
  },

  "pipeline_kwargs": {
    "chunk_length_s": 30,
    "batch_size": 8
  }
}
```

</details>

### Train

Follow these steps to set up and start the training process:

1) **Download Data:**
   - Ensure you have the required datasets by downloading them from [tbd](https://github.zhaw.ch/mict/chall_e2e_stt)
2) **Setup Environment**
    - [Setup environment](#getting-started)
    - Open working directory
      ```shell
      # Note: depending on docker build command
      cd /workspace
      ```
    - Log in to WandB to enable performance tracking and visualization of your training runs. Ensure you have an active account and your project is set up correctly.
      ```
      wandb login
      ```
    - Set Huggingface Token to access private Dataset
      ```
      export HF_TOKEN=your_hf_token
      ```
3) **Configuration:**
   - Set up your training configuration by defining parameters in `config/train/config.json`
   - Use the provided configuration to replicate the results from the paper.
   - Or define your own configuration:
     - `experiment_name`: The name of the experiment (e.g., "chall").
     - `experiment_tag`: A tag to identify the experiment setup (e.g., "wav2vec2_xlsr_1b").
     - `alt_base_path`: Alternative base path for data (e.g., "/work_space_data/chall").
     - `train_corpora`:  The dataset and split used for training (e.g., [["mict-zhaw/chall", ["train"]]]).
     - `eval_corpora`: The dataset and split used for evaluation (e.g., [["mict-zhaw/chall", ["eval"]]]).
     - `data_dir`: Directory where data is stored.
     - `dataset_kwargs`: Additional dataset-specific arguments, See [model card](https://huggingface.co/datasets/mict-zhaw/chall)
     - `seed`: Random seed for reproducibility
     - `device`: Device to run the training on (e.g., "cuda").
     - `wav2vec_base_model`: Base model for wav2vec (e.g., "facebook/wav2vec2-xls-r-300m").
     - `checkpoint` (Optional): Path to a checkpoint to resume training (null if not resuming).
     - `n_steps`: Number of training steps (e.g., 250).
     - `batch_size`: Batch size for training (e.g., 4). 
     - `eval_batch_size`: Batch size for evaluation (e.g., 1).
     - `gradient_accumulation_steps`: Number of steps to accumulate gradients (e.g., 80).
     - `gradient_checkpointing`: Whether to use gradient checkpointing.
     - `learning_rate`: Learning rate for training (e.g., 3e-05).
     - `label_smoothing_factor`: Factor for label smoothing (e.g., 0.0).
     - `fp16`: Use 16-bit floating point precision.
     - `bf16`: Use bfloat16 precision. If you have the right hardware, use BF16
     - `optim`: Optimizer to use (e.g., "adamw_bnb_8bit").
     - `freeze_w2vm`: Whether to freeze the wav2vec model (false).
     - `save_total_limit`: Maximum number of checkpoints to save (e.g., 2).
     - `validation_freq`: Frequency of validation steps (e.g., every 100 steps).
     - `save_step`: Steps interval for saving checkpoints (e.g., every 100 steps).
     - `logging_step`: Steps interval for logging (e.g., every 20 steps).
     - `n_train_samples`: Number of training samples to use (-1 for all).
     - `n_valid_samples`: Number of validation samples to use (-1 for all).

4) **Launch Training**: 
   - **Start Train**: Do this before running distributed to check if everything is setup correctly.
 
     ```shell
     python train.py
     ```
     
   - **Start Distributed**: To start training, execute the following command (The model for ACL was trained on six NVIDIA Tesla V100 GPUs).
   Adjusting the number of GPUs (nproc_per_node) as necessary, but make sure that `num_gpus` x `batch_size` x `gradient_accumulation_steps` is approximately 1280.
      
     ```shell
     python -m torch.distributed.launch --nproc_per_node 6 train.py
     ```

### Create Hypothesis
tbd


### Create Alignments
tbd

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Citation
If you use the dataset or code in your research, please kindly cite our work:

```bibtex
@tbd{
    title = "Error-preserving Automatic Speech Recognition of Young English Learners’ Language",
}
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>