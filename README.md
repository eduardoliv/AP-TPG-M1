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

### Repository Organization

```md

/ (Root)
├── tarefa_1/
│   ├── clean_input_datasets               (Cleaned and uniformized input datasets)
│   ├── clean_output_datasets              (Cleaned and uniformized output datasets)
│   ├── original_input_datasets            (Original input datasets)
│   ├── original_output_datasets           (Original output datasets)
│   └── clean_pipeline.py                  (Script for cleaning and normalizing datasets)
├── tarefa_2/
│   ├── classify_input_datasets            (Input datasets for classification)
│   ├── classify_output_datasets           (Output datasets for classified results)
│   ├── logistic_regression_model.ipynb    (Notebook for the Logistic Regression model)
│   ├── dnn_model.ipynb                    (Notebook for the Deep Neural Network model)
│   ├── rnn_model.py                       (Recurrent Neural Network model)
│   ├── lr_model_weights/                  (Logistic Regression model weights)
│   ├── dnn_model_weights/                 (Deep Neural Network model weights)
│   ├── models/                            (Code models used in the notebooks)
│   │   ├── dnn_model.py
│   │   └── logistic_regression_model.py
│   └── helpers/                           (Utility modules for the models)
├── tarefa_3/
│   └── Work-in-Progress                   (Under development)
└── tarefa_4/
    └── Work-in-Progress                   (Under development)

```

### Code Dependencies

#### Using pip

```bash

pip install pandas chardet scipy nltk

```

#### Using conda

```bash

conda install -c conda-forge pandas chardet scipy nltk

```