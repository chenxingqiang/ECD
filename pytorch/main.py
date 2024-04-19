import scipy.io as sio
import torch
import time
from pytorch import *
import matplotlib.pyplot as plt
import numpy as np


def load_synthetic_dataset(file_path):
    """
    Load synthetic network dataset from a .mat file.
    Args:
        file_path (str): Path to the .mat file containing the synthetic dataset
    Returns:
        W_Cube (list): List of adjacency matrices for each timestep
        GT_Matrix (torch.Tensor): Ground truth community assignments
        for each node at each timestep
    """
    data = sio.loadmat(file_path)
    W_Cube = data['W_Cube'][0]
    GT_Matrix = torch.tensor(data['GT_Matrix']).float()
    return W_Cube, GT_Matrix

def load_realworld_dataset(data_file, gt_file):
    """
    Load real-world network dataset from .mat files.
    Args:
        data_file (str): Path to the .mat file containing the real-world dataset
        gt_file (str): Path to the .mat file containing the ground truth communities
    Returns:
        W_Cube (list): List of adjacency matrices for each timestep
        GT_Cube (list): List of ground truth community assignments for each timestep
    """
    data = sio.loadmat(data_file)
    W_Cube = data['W_Cube'][0]
    GT_Cube = sio.loadmat(gt_file)['dynMoeaResult'][0]
    return W_Cube, GT_Cube


def save_results(avg_dynMod, avg_dynNmi, output_file="results.npz"):
    """
    Save the average modularity and NMI results to a file.

    Args:
        avg_dynMod (torch.Tensor): Average dynamic modularity of shape (num_timestep,)
        avg_dynNmi (torch.Tensor): Average dynamic NMI of shape (num_timestep,)
        output_file (str): Output file name (default: 'results.npz')
    """
    np.savez(output_file, avg_dynMod=avg_dynMod.numpy(), avg_dynNmi=avg_dynNmi.numpy())
    print(f"Results saved to {output_file}")


def display_results(avg_dynMod, avg_dynNmi):
    """
    Display the average modularity and NMI results using plots.

    Args:
        avg_dynMod (torch.Tensor): Average dynamic modularity of shape (num_timestep,)
        avg_dynNmi (torch.Tensor): Average dynamic NMI of shape (num_timestep,)
    """
    num_timestep = len(avg_dynMod)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_timestep + 1), avg_dynMod, marker="o")
    plt.xlabel("Timestep")
    plt.ylabel("Average Modularity")
    plt.title("Average Dynamic Modularity")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_timestep + 1), avg_dynNmi, marker="o")
    plt.xlabel("Timestep")
    plt.ylabel("Average NMI")
    plt.title("Average Dynamic NMI")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    # Load dataset
    flag = 2  # Set flag = 1 for synthetic networks or flag = 2 for real-world networks
    if flag == 1:
        # Synthetic networks
        dataset_name = 'syn_fix_3'  # Choose from: 'syn_fix_3', 'syn_fix_5', 'syn_var_3', 'syn_var_5', 'expand', 'mergesplit'
        file_path = f"./ECD/datasets/{dataset_name}.mat"
        W_Cube, GT_Matrix = load_synthetic_dataset(file_path)
    elif flag == 2:
        # Real-world networks
        dataset_name = 'enron'  # Choose from: 'cell', 'enron'
        data_file = f"./ECD/datasets/{dataset_name}.mat"
        gt_file = f"./ECD/datasets/firststep_DYNMOGA_{dataset_name}.mat"
        W_Cube, GT_Cube = load_realworld_dataset(data_file, gt_file)
    else:
        raise ValueError("Invalid flag value. Choose 1 for synthetic networks or 2 for real-world networks.")

    # Set parameters
    maxgen = 100  # Maximum number of iterations
    pop_size = 100  # Population size
    num_neighbor = 10  # Neighbor size for each subproblem
    p_mutation = 0.2  # Mutation rate
    p_migration = 0.5  # Migration rate
    p_mu_mi = 0.5  # Parameter to control the execution of mutation and migration
    Threshold = 0.8  # R=1-Threshold is the parameter related to population generation
    num_repeat = 5  # Number of repeated runs

    # Initialize result arrays
    num_timestep = len(W_Cube)
    dynMod = torch.zeros(num_timestep, num_repeat)
    dynNmi = torch.zeros(num_timestep, num_repeat)
    dynPop = [None] * num_timestep
    dynTime = torch.zeros(num_timestep, num_repeat)
    ECD_Result = [None] * num_timestep

    for r in range(num_repeat):
        start_time = time.time()

        # Run ECD_1 for the first timestep
        dynMod[0, r], dynPop[0], ECD_Result[0], dynTime[0, r] = ECD_1(
            W_Cube[0], maxgen, pop_size, p_mutation, p_migration, p_mu_mi, Threshold
        )

        # Calculate NMI for the first timestep
        dynNmi[0, r] = nmi(ECD_Result[0], GT_Cube[0])

        print(
            f"timestep = 1, Modularity = {dynMod[0, r]:.4f}, NMI = {dynNmi[0, r]:.4f}"
        )

        for timestep_num in range(1, num_timestep):
            # Run ECD_2 for the following timesteps
            (
                dynMod[timestep_num, r],
                dynPop[timestep_num],
                ECD_Result[timestep_num],
                dynTime[timestep_num, r],
            ) = ECD_2(
                torch.from_numpy(W_Cube[timestep_num]),
                maxgen,
                pop_size,
                p_mutation,
                p_migration,
                p_mu_mi,
                num_neighbor,
                ECD_Result[timestep_num - 1],
                Threshold,
            )

            # Calculate NMI for the following timesteps
            dynNmi[timestep_num, r] = nmi(
                ECD_Result[timestep_num], GT_Cube[timestep_num]
            )

            print(
                f"timestep = {timestep_num + 1},
                Modularity = {dynMod[timestep_num, r]:.4f},
                NMI = {dynNmi[timestep_num, r]:.4f}"
            )

    # Calculate average modularity and NMI
    avg_dynMod = dynMod.mean(dim=1)
    avg_dynNmi = dynNmi.mean(dim=1)

    # Save and display results
    save_results(avg_dynMod, avg_dynNmi, output_file="decs_results.npz")
    display_results(avg_dynMod, avg_dynNmi)


if __name__ == "__main__":
    main()