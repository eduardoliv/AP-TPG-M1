#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: miguelrocha
(Adapted by: Grupo 03)
"""

"""
Usage example:

TRAIN:
python rnn_model.py --input_csv ../tarefa_1/clean_input_datasets/dataset1_inputs.csv --output_csv ../tarefa_1/clean_output_datasets/dataset1_outputs.csv --train rnn_modelo.pkl --epochs 50 --batch_size 8 --learning_rate 0.01 --momentum 0.9 --bptt_trunc 2 --verbose True
python rnn_model.py --input_csv ../tarefa_1/clean_input_datasets/dataset1_inputs.csv --output_csv ../tarefa_1/clean_output_datasets/dataset1_outputs.csv --train 

TEST:
python rnn_model.py --input_csv ../tarefa_1/clean_input_datasets/gpt_vs_human_data_set_inputs.csv --output_csv ../tarefa_1/clean_output_datasets/gpt_vs_human_data_set_outputs.csv --test rnn_modelo.pkl

python rnn_model.py --input_csv ../tarefa_1/clean_input_datasets/dataset2_inputs.csv --output_csv ../tarefa_1/clean_output_datasets/dataset2_outputs.csv --test rnn_model3.pkl
"""

from copy import deepcopy
from typing import Tuple

import numpy as np
import argparse
import pickle
import os
import re

from helpers.activation import TanhActivation, ActivationLayer
from helpers.layers import Layer
from helpers.losses import BinaryCrossEntropy, LossFunction
from helpers.optimizer import Optimizer
from helpers.metrics import accuracy
from helpers.dataset import Dataset
# from sklearn.metrics import f1_score

def convert_text_to_sequences(df, text_col="Text"):
    texts = df[text_col].astype(str).tolist()
    vocab = {}
    token_lists = []
    max_len = 0
    for txt in texts:
        tokens = txt.lower().split()
        token_lists.append(tokens)
        if len(tokens) > max_len:
            max_len = len(tokens)
    # Build vocab
    for tokens in token_lists:
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab) + 1
    return token_lists, vocab, max_len

def pad_or_truncate(token_lists, vocab, max_len):
    sequences = []
    for tokens in token_lists:
        seq = []
        for token in tokens:
            seq.append(vocab[token])
        # truncate
        seq = seq[:max_len]
        # pad
        seq += [0]*(max_len - len(seq))
        sequences.append(seq)
    return np.array(sequences)

class RNN(Layer):
    """A Vanilla Fully-Connected Recurrent Neural Network layer."""

    def __init__(self, n_units: int, activation: ActivationLayer = TanhActivation(), bptt_trunc: int = 5, input_shape: Tuple = None, epochs = 100, batch_size = 128, learning_rate = 0.01, momentum = 0.90, loss: LossFunction = BinaryCrossEntropy, metric:callable = accuracy, verbose = False):
        """
        Initializes the layer.

        Parameters
        ----------
        n_units: int
            The number of units in the layer (i.e. the number of hidden states).
        activation: ActivationLayer
            The activation function to apply to the output of each state.
        bptt_trunc: int
            The number of time steps to backpropagate through time (i.e. the number of time steps to unroll the RNN).
        input_shape: Tuple
            The shape of the input to the layer.
        """
        self.input_shape = input_shape
        self.n_units = n_units
        self.activation = activation
        self.bptt_trunc = bptt_trunc
        self.metric = metric
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.loss_fn = loss()
        self.metric_fn = metric
        self.verbose = verbose

        # Weights
        self.W = None  # Weight of the previous state
        self.V = None  # Weight of the output
        self.U = None  # Weight of the input

        # Optimizers
        self.U_opt = None
        self.V_opt = None
        self.W_opt = None

        # For tracking training
        self.history = {}

    def initialize(self, optimizer):
        """
        Initializes the weights of the layer.

        Parameters
        ----------
        optimizer: Optimizer
            The optimizer to use for updating the weights.
        """
        timesteps, input_dim = self.input_shape
        # Initialize the weights
        limit = 1 / np.sqrt(input_dim)
        self.U = np.random.uniform(-limit, limit, (self.n_units, input_dim))
        limit = 1 / np.sqrt(self.n_units)
        self.V = np.random.uniform(-limit, limit, (input_dim, self.n_units))
        self.W = np.random.uniform(-limit, limit, (self.n_units, self.n_units))
        # Weight optimizers
        self.U_opt = deepcopy(optimizer)
        self.V_opt = deepcopy(optimizer)
        self.W_opt = deepcopy(optimizer)

    def get_mini_batches(self, X, y = None,shuffle = True):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        self.batch_size = min(self.batch_size, n_samples)  # Ajusta batch_size dinamicamente
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, n_samples - self.batch_size + 1, self.batch_size):
            if y is not None:
                yield X[indices[start:start + self.batch_size]], y[indices[start:start + self.batch_size]]
            else:
                yield X[indices[start:start + self.batch_size]], None
    
    def forward_propagation(self, input: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Perform forward propagation on the given input.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.
        training: bool
            Whether the layer is in training mode or in inference mode.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        self.layer_input = input
        batch_size, timesteps, input_dim = input.shape

        # Save these values for use in backprop.
        self.state_input = np.zeros((batch_size, timesteps, self.n_units))
        self.states = np.zeros((batch_size, timesteps + 1, self.n_units))
        self.outputs = np.zeros((batch_size, timesteps, input_dim))

        # Set last time step to zero for calculation of the state_input at time step zero (already zero?)
        # self.states[:, -1] = np.zeros((batch_size, self.n_units))
        for t in range(timesteps):
            # Input to state_t is the current input and output of previous states
            self.state_input[:, t] = input[:, t].dot(self.U.T) + self.states[:, t - 1].dot(self.W.T)
            self.states[:, t] = self.activation.activation_function(self.state_input[:, t])
            self.outputs[:, t] = self.states[:, t].dot(self.V.T)

        return self.outputs

    def backward_propagation(self, accum_grad: np.ndarray) -> np.ndarray:
        """
        Perform backward propagation on the given output error.

        Parameters
        ----------
        accum_grad: numpy.ndarray
            The accumulated gradient from the previous layer.
        Returns:
        --------
        numpy.ndarray
            The accumulated gradient w.r.t the input of the layer.
        """
        _, timesteps, _ = accum_grad.shape

        # Variables where we save the accumulated gradient w.r.t each parameter
        grad_U = np.zeros_like(self.U)
        grad_V = np.zeros_like(self.V)
        grad_W = np.zeros_like(self.W)
        # The gradient w.r.t the layer input.
        # Will be passed on to the previous layer in the network
        accum_grad_next = np.zeros_like(accum_grad)

        # Back Propagation Through Time
        for t in reversed(range(timesteps)):
            # Update gradient w.r.t V at time step t
            grad_V += accum_grad[:, t].T.dot(self.states[:, t])
            # Calculate the gradient w.r.t the state input
            grad_wrt_state = accum_grad[:, t].dot(self.V) * self.activation.derivative(self.state_input[:, t])
            # Gradient w.r.t the layer input
            accum_grad_next[:, t] = grad_wrt_state.dot(self.U)
            # Update gradient w.r.t W and U by backprop. from time step t for at most
            # self.bptt_trunc number of time steps
            for t_ in reversed(np.arange(max(0, t - self.bptt_trunc), t + 1)):
                grad_U += grad_wrt_state.T.dot(self.layer_input[:, t_])
                grad_W += grad_wrt_state.T.dot(self.states[:, t_ - 1])
                # Calculate gradient w.r.t previous state
                grad_wrt_state = grad_wrt_state.dot(self.W) * self.activation.derivative(self.state_input[:, t_ - 1])

        # Update weights
        self.U = self.U_opt.update(self.U, grad_U)
        self.V = self.V_opt.update(self.V, grad_V)
        self.W = self.W_opt.update(self.W, grad_W)

        return accum_grad_next

    def output_shape(self) -> tuple:
        """
        Returns the shape of the output of the layer.

        Returns
        -------
        tuple
            The shape of the output of the layer.
        """
        return self.input_shape

    def parameters(self) -> int:
        """
        Returns the number of parameters of the layer.

        Returns
        -------
        int
            The number of parameters of the layer.
        """
        return np.prod(self.W.shape) + np.prod(self.U.shape) + np.prod(self.V.shape)
    
    def predict_proba(self, dataset):
        """
        Return predicted probabilities at the last time step.
        dataset shape: (num_samples, timesteps, input_dim)
        Returns shape: (num_samples,)
        """
        out = self.forward_propagation(dataset, training=False)
        # last time step, single dimension
        return out[:, -1, 0]

    def predict(self, dataset):
        """
        Return binary predictions (0 or 1).
        """
        probs = self.predict_proba(dataset)
        return (probs >= 0.5).astype(float)

    def score(self, dataset, predictions):
        if self.metric is not None:
            return self.metric(dataset.Y, predictions)
        else:
            raise ValueError("No metric specified for the neural network.")

    def fit(self, dataset):
        X = dataset.X
        y = dataset.Y

        if np.ndim(y) == 1:
            y = np.expand_dims(y, axis=1)

        self.history = {}
        
        for epoch in range(1, self.epochs + 1):
            epoch_preds = []
            epoch_labels = []

            for X_batch, y_batch in self.get_mini_batches(X, y):
                # forward => (batch_size, timesteps, input_dim)
                out = self.forward_propagation(X_batch, training=True)
                # extract final time step => shape (batch_size,)
                pred = out[:, -1, 0]

                # compute derivative wrt final step
                d_pred = self.loss_fn.derivative(y_batch.ravel(), pred)
                # embed into full shape
                grad_out = np.zeros_like(out)          # (batch_size, timesteps, input_dim)
                grad_out[:, -1, 0] = d_pred            # place derivative at final step

                # backward
                self.backward_propagation(grad_out)

                # store predictions
                epoch_preds.append(pred)
                epoch_labels.append(y_batch.ravel())

            # after mini-batches => compute epoch-level metrics
            preds_all = np.concatenate(epoch_preds)      # shape => (n_samples,)
            labels_all = np.concatenate(epoch_labels)    # shape => (n_samples,)

            # compute loss
            loss_val = self.loss_fn.loss(labels_all, preds_all)

            if self.metric_fn is not None:
                metric_val = self.metric_fn(labels_all, preds_all)
            else:
                metric_val = None

            self.history[epoch] = {"loss": loss_val, "metric": metric_val}
            if self.verbose:
                m_s = f"{metric_val:.4f}" if metric_val is not None else "NA"
                print(f"Epoch {epoch}/{self.epochs} - loss: {loss_val:.4f} - metric: {m_s}")

        return self

###################################################

# Lista de hiperparâmetros para testar
HYPERPARAMS = [
    {"epochs": 5, "batch_size": 8, "learning_rate": 0.01, "momentum": 0.9, "bptt_trunc": 2},
    {"epochs": 10, "batch_size": 16, "learning_rate": 0.005, "momentum": 0.95, "bptt_trunc": 3},
    {"epochs": 7, "batch_size": 8, "learning_rate": 0.007, "momentum": 0.8, "bptt_trunc": 2},
    {"epochs": 6, "batch_size": 16, "learning_rate": 0.01, "momentum": 0.85, "bptt_trunc": 3},
    {"epochs": 8, "batch_size": 12, "learning_rate": 0.008, "momentum": 0.9, "bptt_trunc": 4}
]

def registar_modelo(info, model_filename):
    """Registra ou atualiza informações do modelo no log."""
    log_file = "rnn_log_modelos.txt"
    modelo_inicio = f"MODELO: {model_filename}\n"
    modelo_fim = "-"*50 + "\n"
    
    # Verifica se o modelo já está no log
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            linhas = f.readlines()
        
        if modelo_inicio in linhas:
            # Se o modelo já existe, adiciona na seção correta
            novo_conteudo = []
            dentro_modelo = False
            for linha in linhas:
                if linha.startswith("MODELO:"):
                    dentro_modelo = (linha == modelo_inicio)
                if dentro_modelo and linha.startswith("TESTES"):
                    novo_conteudo.append(linha)
                    novo_conteudo.append(info + "\n")
                    dentro_modelo = False  # Garante que só adicionamos uma vez
                    continue
                novo_conteudo.append(linha)
            with open(log_file, "w", encoding="utf-8") as f:
                f.writelines(novo_conteudo)
            return
    
    # Se o modelo não está no log, cria uma nova entrada
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(modelo_inicio)
        f.write("\nCOMO FOI O TREINO\n")
        f.write(info + "\n")
        f.write("\nTESTES\n")
        f.write(modelo_fim)

# Ajuste na função get_mini_batches para evitar erro de índice
def get_mini_batches2(X, y, batch_size, shuffle=True):
    n_samples = X.shape[0]
    batch_size = min(batch_size, n_samples)  # Garante que batch_size nunca seja maior que n_samples
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        yield X[indices[start:end]], y[indices[start:end]]

# Função para treinar e testar modelos
def train_and_evaluate(input_csv, output_csv):
    train_ds, vocab, max_len = Dataset.create_train_dataset(
        train_input_csv=input_csv, train_output_csv=output_csv, max_len=None, sep="\t"
    )
    
    X_train, y_train, X_val, y_val = train_ds.train_test_split(p=0.7)
    best_models = []
    
    for i, params in enumerate(HYPERPARAMS):
        print(f"Treinando modelo {i+1} com hiperparâmetros: {params}")
        
        batch_size = min(params["batch_size"], X_train.shape[0])
        
        rnn = RNN(
            n_units=4,
            activation=TanhActivation(),
            bptt_trunc=params["bptt_trunc"],
            input_shape=(max_len, 1),
            epochs=params["epochs"],
            batch_size=batch_size,
            learning_rate=params["learning_rate"],
            momentum=params["momentum"],
            loss=BinaryCrossEntropy,
            metric=accuracy,
            verbose=False
        )
        
        optimizer = Optimizer(learning_rate=params["learning_rate"], momentum=params["momentum"])
        rnn.initialize(optimizer)
        
        for X_batch, y_batch in get_mini_batches2(X_train, y_train, batch_size):
            rnn.fit(Dataset(X=X_batch, Y=y_batch))
        
        preds = rnn.predict(X_val)
        acc = accuracy(y_val, preds)
        # f1 = f1_score(y_val, preds)
        # print(f"Modelo {i+1} - Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
        
        best_models.append((acc, f1, rnn, params))
    
    best_models.sort(reverse=True, key=lambda x: x[0])
    
    for i, (acc, f1, model, params) in enumerate(best_models[:3]):

        # Encontrar o maior índice existente de modelos salvos
        existing_models = [f for f in os.listdir() if re.match(r"rnn_model\d+\.pkl", f)]
        if existing_models:
            max_index = max(int(re.search(r"rnn_model(\d+)\.pkl", f).group(1)) for f in existing_models)
        else:
            max_index = 0

        model_filename = f"rnn_model{max_index + 1}.pkl"

        with open(model_filename, "wb") as f:
            pickle.dump({"rnn": model, "vocab": vocab, "max_len": max_len, "params": params}, f)

        log_info = (f"Hiperparâmetros: {params}\n"
                    f"Dataset de treino: {input_csv}\n"
                    f"Accuracy (Validação): {acc:.4f}\n"
                    f"F1-score (Validação): {f1:.4f}")
        registar_modelo(log_info, model_filename)
        print(f"Modelo {i+1} registado.")

# Função para testar o modelo
def test_model(input_csv, output_csv, model_filename):
    with open(model_filename, "rb") as f:
        model_data = pickle.load(f)
    
    rnn = model_data["rnn"]
    vocab = model_data.get("vocab", {})
    max_len = model_data.get("max_len", None)
    
    test_ds = Dataset.create_test_dataset(
        test_input_csv=input_csv, test_output_csv=output_csv, vocab=vocab, max_len=max_len, sep="\t"
    )
    
    preds = rnn.predict(test_ds.X)
    acc = accuracy(test_ds.Y, preds)
    # f1 = f1_score(test_ds.Y, preds)

    log_info = (f"Modelo testado: {model_filename}\n"
                f"Dataset de teste: {input_csv}\n"
                f"Accuracy: {acc:.4f}\n"
                # f"F1-score: {f1:.4f}"
                )
    registar_modelo(log_info, model_filename)
    print(f"Teste registrado para {model_filename}.")

###################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="Path to input CSV (ID, Text)")
    parser.add_argument("--output_csv", required=True, help="Path to output CSV (ID, Label)")
    parser.add_argument("--train", action="store_true", help="Treinar o modelo e salvar com este nome")
    parser.add_argument("--test", metavar="model_filename", help="Testar o modelo usando este nome")
    # parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (Default: 100)")
    # parser.add_argument("--batch_size", type=int, default=6, help="Mini-batch size (Default: 6)")
    # parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate (Default: 0.01)")
    # parser.add_argument("--momentum", type=float, default=0.9, help="Momentum (Default: 0.9)")
    # parser.add_argument("--bptt_trunc", type=int, default=1, help="Truncation steps for BPTT (Default: 1)")
    # parser.add_argument("--verbose", default=True, help="Print training details (Default: True)")
    args = parser.parse_args()

    # # Load Datasets
    # train_ds, vocab, max_len = Dataset.create_train_dataset(
    #     train_input_csv=args.input_csv,
    #     train_output_csv=args.output_csv,
    #     max_len=None,
    #     sep="\t"
    # )

    # test_ds = Dataset.create_test_dataset(
    #     test_input_csv=args.input_csv,
    #     test_output_csv=args.output_csv,
    #     vocab=vocab,
    #     max_len=max_len,
    #     sep="\t"
    # )

    # # Instantiate the RNN
    # rnn = RNN(
    #     n_units=4,
    #     activation=TanhActivation(),
    #     bptt_trunc=args.bptt_trunc,
    #     input_shape=(max_len, 1),  # timesteps=max_len, input_dim=1
    #     epochs=args.epochs,
    #     batch_size=args.batch_size,
    #     learning_rate=args.learning_rate,
    #     momentum=args.momentum,
    #     loss=BinaryCrossEntropy,
    #     metric=accuracy,
    #     verbose=args.verbose
    # )

    # # Initialize the RNN with an SGD optimizer
    # optimizer = Optimizer(learning_rate=args.learning_rate, momentum=args.momentum)
    # rnn.initialize(optimizer)

    # # Fit the RNN using dataset
    # rnn.fit(train_ds)

    # # Evaluate final performance
    # preds = rnn.predict(test_ds.X)
    # score = rnn.score(test_ds, preds)
    # print(f"Final score: {score:.4f}")

    if args.train:
        train_and_evaluate(args.input_csv, args.output_csv)
    elif args.test:
        test_model(args.input_csv, args.output_csv, args.test)
    else:
        print("Especifique --train ou --test para rodar o script.")

if __name__ == "__main__":
    main()
