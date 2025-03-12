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
│   ├── classify_output_datasets            (Classified output datasets)
│   ├── clean_input_datasets                (Uniformized input datasets)
│   ├── clean_output_datasets               (Uniformized output datasets)
│   ├── original_input_datasets             (Original input datasets)
│   ├── original_output_datasets            (Original output datasets)
│   └── clean_pipeline.py                   (Code for cleaning and normalizing datasets)
├── tarefa_2/
│   ├── logistic_regression_model.ipynb     (Logistic Regression Model Notebook)
│   ├── dnn_model.py                        (Deep Neural Network Model)
│   ├── rnn_model.py                        (Recurrent Neural Network Model)
│   ├── models/                             (Code Models used on the Jupyter Notebooks)
|   |   └── logistic_regression_model.py
│   └── helpers/                            (Code Utilities used on the Models)
│       ├── activation.py
|       ├── dataset.py
│       ├── enums.py
│       ├── layers.py
│       ├── losses.py
│       ├── metrics.py
│       ├── models.py
│       ├── optimizer.py
│       └── regularizer.py
├── tarefa_3/
|   └── Work-in-Progress
└── tarefa_4/
    └── Work-in-Progress

```

### Code Dependencies

#### Using pip

```bash

pip install pandas chardet scipy

```

#### Using conda

```bash

conda install -c conda-forge pandas chardet scipy

```