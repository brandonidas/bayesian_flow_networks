# Copyright (C) 2023 Maxime Robeyns <dev@maximerobeyns.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Swiss roll example using a simple continuous BFN"""

import os
import numpy as np
import torch as t
import matplotlib.pyplot as plt
import pickle

from typing import Callable, Tuple
from torchtyping import TensorType as Tensor
from sklearn.datasets import make_swiss_roll
from torch.utils.data import DataLoader, TensorDataset, random_split
from simple_nn import SimpleNN

from torch_bfn import ContinuousBFN, LinearNetwork
from torch_bfn.utils import EMA, norm_denorm, str_to_torch_dtype

def make_smiles_dset(
    n: int, bs: int = 512, noise: float = 0, dtype: t.dtype = t.float32
) -> Tuple[
    DataLoader, DataLoader, Callable[[Tensor["B", "D"]], Tensor["B", "D"]]
]:
    with open("smaller_bfn_exercise.pkl", 'rb') as file:
        data = pickle.load(file)
    
    # Create a normalised 'swiss roll' dataset
    X_np = np.array(data) # assume for now its tuples

    emb_smiles_arrays = [np.array(row['emb_smiles']) for row in data]
    # Converting list of np.arrays into a single 2D np.array
    matrix = np.vstack(emb_smiles_arrays)
    logp_values = [row['logp'] for row in data] 
    logp_array = np.array(logp_values).reshape(-1, 1)
    matrix = np.concatenate((matrix, logp_array), axis=1)
    print(matrix.shape)  # should print (5000, <length_of_emb_smiles_array>)
    
    X_np = matrix  # todo verify which indices are smiles_emb
    
    X = t.tensor(X_np, dtype=dtype)
    
    X, denorm = norm_denorm(X)

    dset = TensorDataset(X)
    train_size = len(dset) - bs
    val_size = len(dset) - train_size

    train_dset, val_dset = random_split(dset, [train_size, val_size])
    
    train_loader = DataLoader(train_dset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dset, batch_size=bs, shuffle=False)
    
    return train_loader, val_loader, denorm

def train(
    model: ContinuousBFN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    denorm: Callable[[Tensor["B", "D"]], Tensor["B", "D"]],
    epochs: int = 10000,
    device_str: str = "cpu",
    dtype_str: str = "float32",
):
    # load regressor model first
    regressor_model = SimpleNN(256)
    state_dict = t.load('simple_nn_model.pth', map_location=t.device('cpu'))
    regressor_model.load_state_dict(state_dict)
    regressor_model.eval()

    device = t.device(device_str)
    dtype = str_to_torch_dtype(dtype_str)
    ema = EMA(0.9)

    model.to(device, dtype)
    opt = t.optim.Adam(model.parameters(), lr=1e-4)
    ema.register(model)

    for epoch in range(epochs):
        loss = None
        for batch in train_loader:
            X = batch[0].to(device, dtype)
            loss = model.loss(X, sigma_1=0.01).mean()
            # loss = model.discrete_loss(X, sigma_1=0.01, n=30).mean()
            opt.zero_grad()
            loss.backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)

        if epoch % 100 == 0:
            assert loss is not None
            print(loss.item())
            samples = model.sample(100, sigma_1=0.01, n_timesteps=100)
            # print(samples.shape) torch.Size([100, 257])
            with t.no_grad(): 
                samples_features = samples[:, :256]
                samples_targets = samples[:,-1]
                sample_predictions = regressor_model(samples_features)
                mse = regressor_model.criterion(sample_predictions, samples_targets)
                print("        training MSE on logp:" + str(mse))
            
    t.save(model.state_dict(), 'smiles_bfn_model.pth')

if __name__ == "__main__":

    train_loader, val_loader, denorm = make_smiles_dset(int(1e4))
    device = "cpu"
    dtype = "float32"

    net = LinearNetwork(
        dim=257,
        hidden_dims=[128, 128],
        sin_dim=16,
        time_dim=64,
        random_time_emb=False,
        dropout_p=0.0,
    )

    print("DEVICE: " + str(device))
    model = ContinuousBFN(
        dim=257,
        net=net,
        device_str=device,
        dtype_str=dtype,
    )

    train(
        model,
        train_loader,
        val_loader,
        denorm,
        device_str=device,
        dtype_str=dtype,
    )