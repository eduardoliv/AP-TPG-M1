# Deep Learning – Module 1

## Group Practical Assignment

### Group Composition

| #id     | Name                             | Email                    | Group |
| ------- | -------------------------------- | ------------------------ | ----- |
| pg57890 | Mateus Lemos Martins             | pg57890@alunos.uminho.pt | 03    |
| pg53802 | Emanuel Lopes Monteiro da Silva  | pg53802@alunos.uminho.pt | 03    |
| pg57879 | João Andrade Rodrigues           | pg57879@alunos.uminho.pt | 03    |
| pg52688 | Jorge Eduardo Quinteiro Oliveira | pg52688@alunos.uminho.pt | 03    |
| e12338  | Bernardo Dutra Lemos             | e12338@alunos.uminho.pt  | 03    |

---

### Phase Submissions

#### Delivery 2 (24/03/2025)

Submission 2 is based on the results of the BERT and RoBERTa LLM models.
Below is the structure of these models, as well as the generated results for the submission. 

```md
/ (Root)
└── Submissao2/
    ├── classify_input_datasets/
    ├── classify_output_datasets/
    │   ├── dataset3_outputs_llm_bert_model-s1.csv        **Dataset** generated from BERT LLM model for Submission 2
    │   └── dataset3_outputs_llm_roberta_model-s2.csv     **Dataset** generated from RoBERTa LLM model for Submission 2
    ├── llm_bert_model-s1.ipynb                           **Notebook** for the BERT LLM model for Submission 2
    ├── llm_roberta_model-s2.ipynb                        **Notebook** for the RoBERTa LLM model for Submission 2
    ├──  llm_bert_model_weights/
    └──  llm_roberta_model_weights/
```

#### Delivery 1 (17/03/2025)

Submission 1 is based on the results of the DNN and RNN models.
Below is the structure of these models, as well as the generated results for the submission. 

```md
/ (Root)
└── Submissao1/
    ├── classify_input_datasets/
    ├── classify_output_datasets/
    │   ├── dataset2_outputs_rnn_model.csv     **Dataset** generated from RNN model for Submission 1
    │   └── dataset2_outputs_dnn_model.csv     **Dataset** generated from DNN model for Submission 1
    ├── dnn_model.ipynb                        **Notebook** for the Deep Neural Network model for Submission 1
    ├── rnn_model.ipynb                        **Notebook** for the Recurrent Neural Network model for Submission 1
    ├── dnn_model_weights/         
    ├── models/
    │   ├── dnn_model.py
    │   └── rnn_model.py
    └── helpers/
```

---

### Repository Organization

```md
/ (Root)
├── Submissao1/                                # Submission 1 delivery models and generated output datasets
├── Submissao2/                                # Submission 2 delivery models and generated output datasets
└── Tarefas/
    ├── tarefa_1/
    │   ├── clean_input_datasets/              # Cleaned and uniformized input datasets
    │   ├── clean_output_datasets/             # Cleaned and uniformized output datasets
    │   ├── original_input_datasets/           # Original input datasets
    │   ├── original_output_dataset/           # Original output datasets
    │   └── clean_pipeline.py                  # Script for cleaning and normalizing datasets
    ├── tarefa_2/
    │   ├── classify_input_datasets            # Input datasets for classification
    │   ├── classify_output_datasets           # Output datasets for classified results
    │   ├── logistic_regression_model.ipynb    # Notebook for the Logistic Regression model
    │   ├── dnn_model.ipynb                    # Notebook for the Deep Neural Network model
    │   ├── rnn_model.ipynb                    # Notebook for the Recurrent Neural Network model
    │   ├── lr_model_weights/                  # Logistic Regression model weights
    │   ├── dnn_model_weights/                 # Deep Neural Network model weights
    │   ├── models/                            # Code models used in the notebooks
    │   │   ├── dnn_model.py
    |   |   ├── rnn_model.py
    │   │   └── logistic_regression_model.py
    │   └── helpers/                           # Utility modules for the models
    ├── tarefa_3/
    │   ├── classify_input_datasets            # Input datasets for classification
    │   ├── classify_output_datasets           # Output datasets for classified results
    │   ├── gru_model_weights/                 # Gated Recurrent Units (GRU) Model weights
    │   ├── gru_model.ipynb                    # Notebook for the Gated Recurrent Units (GRU) Model
    │   ├── llm_bert_model_weights/            # Bidirectional Encoder Representations from Transformers (BERT) Model weights
    │   ├── llm_bert_model.ipynb               # Notebook for the Bidirectional Encoder Representations from Transformers (BERT) Model
    │   ├── llm_roberta_model_weights/         # Robustly Optimized BERT Pretraining Approach (RoBERTa) Model weights
    │   ├── llm_roberta_model.ipynb            # Notebook for the Robustly Optimized BERT Pretraining Approach (RoBERTa) Model
    │   ├── lstm_model_weights/                # Long Short-Term Memory (LSTM) Model weights
    │   └── lstm_model.ipynb                   # Notebook for the Long Short-Term Memory (LSTM) Model
    └── tarefa_4/
        └── Work-in-Progress
```

---

### Code Dependencies

#### [CPU] Using pip

```bash
pip install pandas chardet scipy nltk tqdm matplotlib scikit-learn transformers jupyterlab
```

```bash
jupyter lab
```

#### [CPU] Using Conda

```bash
conda create -n ap python=3.10 -y
conda activate ap
conda install -c conda-forge pandas chardet scipy nltk tqdm matplotlib scikit-learn transformers jupyterlab
```

```bash
conda run jupyter-lab
```

#### [NVIDIA GPUs] Using Conda and Tensorflow with CUDA

Validate CUDA and NVCC versions:

```bash
nvidia-smi
```

```bash
nvcc --version
```

Create new Conda env:

```bash
conda create -n tf_gpu python=3.10 -y
conda activate tf_gpu
conda install -c conda-forge cudnn=8.6.0.163 cudatoolkit=11.8
python pip install --upgrade pip
pip install tensorflow==2.10.0
```

If everything is correct, run:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Expected output:

```bash
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

Install the dependencies through pip **(Do not use conda to install dependencies when TensorFlow GPU is installed)**:

```bash
pip install tensorflow==2.10.0 numpy==1.23.5 pandas chardet scipy nltk tqdm matplotlib scikit-learn transformers==4.31.0 notebook
```

Run Jupiter Lab:

```bash
jupyter lab
```

##### Help

In case of error on Tensorflow:

```bash
pip uninstall tensorflow tensorflow-intel -y
pip install tensorflow==2.10.0
```