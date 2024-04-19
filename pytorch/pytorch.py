import math
import time
import numpy as np
import torch


def calculate_q(chromosome, whole_size, num_node, num_edge, adjacent_array):
    """
    calculate the modularity from the genome matrix and chromosome vector, respectively

    Args:
        chromosome: all chromosomes in the population
        whole_size: the total number of the original and new individuals
        num_node: the number of nodes in the network
        num_edge: the number of edges in the network
        adjacent_array: the adjacent matrix of the network

    Returns:
        chromosome: all chromosomes in the population
    """

    # transform the genome matrix into the vector whose elements
    # represent the community to which a node belongs
    node_chrom = change(chromosome, whole_size, num_node)

    for pop_id in range(whole_size):
        num_cluster = torch.max(node_chrom[pop_id]).item()
        e = torch.zeros(num_cluster)
        a = torch.zeros(num_cluster)

        for j in range(num_cluster):
            cluster_id = j
            nodes_in_cluster = torch.where(node_chrom[pop_id] == cluster_id)[0]
            L = len(nodes_in_cluster)

            for k in range(L):
                for m in range(num_node):
                    if adjacent_array[nodes_in_cluster[k], m] == 1:
                        if chromosome[pop_id].genome[nodes_in_cluster[k], m] == 1:
                            e[cluster_id] += 1
                        else:
                            a[cluster_id] += 1

        e /= 2
        a /= 2
        a += e
        e /= num_edge
        a = (a / num_edge) ** 2
        Q = torch.sum(e - a)

        chromosome[pop_id].fitness_1 = Q
        chromosome[pop_id].clusters = node_chrom[pop_id]
        chromosome[pop_id].fitness_2 = modularity(
            adjacent_array, chromosome[pop_id].clusters
        )

    return chromosome


def create_edge_list(adjacent_array):
    """
    Create an edge list from an adjacent matrix

    Args:
        adjacent_array: the adjacent matrix

    Returns:
        edge_begin_end: the edge list
    """
    indices = torch.where(adjacent_array > 0)
    x, y = indices[0], indices[1]
    l = len(x)
    edge_begin_end = torch.zeros(l // 2, 3, dtype=torch.long)
    j = 0

    for i in range(l):
        if x[i] > y[i]:
            edge_begin_end[j, 0] = j
            edge_begin_end[j, 1] = x[i]
            edge_begin_end[j, 2] = y[i]
            j += 1

    return edge_begin_end


def crossover_oneway(chromosome, selected_pop_id, node_num):
    """
    One-way crossover operator

    Args:
        chromosome: all chromosomes in the population
        selected_pop_id: IDs of selected individuals
        node_num: number of nodes

    Returns:
        child_chromosome: the generated child chromosome
    """
    child_chromosome = {
        "genome": None,
        "clusters": None,
        "fitness_1": None,
        "fitness_2": None,
    }

    cross_over_population_id = selected_pop_id
    cross_over_count = torch.randint(1, node_num // 2 + 1, (1,)).item()
    cross_over_node_id = torch.randint(0, node_num, (cross_over_count,))

    # one-way crossover
    child_chromosome["genome"] = chromosome[cross_over_population_id[0]].genome.clone()
    temp2_part = chromosome[cross_over_population_id[1]].genome[cross_over_node_id]
    temp1_whole = chromosome[cross_over_population_id[0]].genome.clone()
    temp1_whole[cross_over_node_id] = temp2_part
    temp1_whole[:, cross_over_node_id] = temp2_part.t()
    child_chromosome["genome"] = temp1_whole

    return child_chromosome


def ECD_1(
    adjacent_array, maxgen, pop_size, p_mutation, p_migration, p_mu_mi, threshold
):
    """
    Detect the community structure at the 1st time step

    Args:
        adjacent_array(Torch.Tensor): the adjacent matrix
        maxgen: the maximum number of iterations
        pop_size: the population size
        p_mutation: the mutation rate
        p_migration: the migration rate
        p_mu_mi: the parameter to control the execution of mutation and migration
        threshold: R=1-threshold is the parameter related to population generation

    Returns:
        Mod: the modularity of the detected community structure
        chromosome: chromosomes in the population
        Division: the detected community structure
        time: running time
    """
    if not torch.equal(adjacent_array, adjacent_array.t()):
        adjacent_array = adjacent_array + adjacent_array.t()

    # set the diagonal elements of an adjacent matrix to be 0
    mask = torch.eye(len(adjacent_array)).bool()
    adjacent_array[mask] = 0

    edge_begin_end = create_edge_list(adjacent_array)

    num_node = adjacent_array.size(1)
    num_edge = torch.sum(adjacent_array) // 2
    adjacent_num = round(0.05 * num_node)
    DynQ = torch.zeros(maxgen)
    children_proportion = 1
    whole_size = math.ceil((1 + children_proportion) * pop_size)

    start_time = time.time()
    chromosome = initial_pnm(
        pop_size, adjacent_array, adjacent_num, num_node, threshold
    )
    chromosome = calculate_q(chromosome, pop_size, num_node, num_edge, adjacent_array)
    chromosome = sort_q(chromosome, pop_size, 1)

    DynQ[0] = chromosome[0].fitness_1
    print(
        f"time_stamp=1; 0 : Q_genome={DynQ[0]:.6f};
        Modularity={chromosome[0].fitness_2:.6f}"
    )

    for i in range(maxgen):
        for pop_id in range(pop_size, whole_size):
            selected_pop_id = []
            while not selected_pop_id or selected_pop_id[0] == selected_pop_id[1]:
                selected_pop_id = torch.randint(0, pop_size, (2,))

            chromosome[pop_id] = crossover_oneway(chromosome, selected_pop_id, num_node)

            if torch.rand(1) < p_mu_mi:
                chromosome[pop_id] = mutation(
                    chromosome[pop_id],
                    p_mutation,
                    num_edge,
                    edge_begin_end
                )
            else:
                chromosome[pop_id] = migration(
                    chromosome[pop_id],
                    num_node,
                    adjacent_array,
                    p_migration
                )

        chromosome = calculate_q(
            chromosome,
            whole_size,
            num_node,
            num_edge,
            adjacent_array
        )
        chromosome = sort_q(chromosome, whole_size, 1)
        chromosome = chromosome[:100]
        DynQ[i] = chromosome[0].fitness_1
        print(
            f"time_stamp=1; {i+1} : Q_genome={DynQ[i]:.6f};
            Modularity={chromosome[0].fitness_2:.6f}"
        )

    Division = chromosome[0].clusters
    Mod = modularity(adjacent_array, chromosome[0].clusters)
    time_used = time.time() - start_time

    return Mod, chromosome, Division, time_used


def ECD_2( adjacent_array,maxgen,pop_size,p_mutation,p_migration,p_mu_mi,num_neighbor,pre_result,threshold):
    """
    Detect the community structure at the time step

    Args:
        adjacent_array(Torch.Tensor ): the adjacent matrix
        maxgen: the maximum number of iterations
        pop_size: the population size
        p_mutation: the mutation rate
        p_migration: the migration rate
        p_mu_mi: the parameter to organize the execution of mutation and migration
        num_neighbor: the neighbor size for each subproblem in decomposition-based
                    multi-objective optimization
        pre_result: the detected community structure at the last time step
        threshold: R=1-threshold is the parameter related to population generation

    Returns:
        Mod: the modularity of the detected community structure
        chromosome: chromosomes in the population
        Division: the detected community structure
        time: running time
    """
    global idealp, weights, neighbors

    if not torch.equal(adjacent_array, adjacent_array.t()):
        adjacent_array = adjacent_array + adjacent_array.t()

    # set the diagonal elements of an adjacent matrix to be 0
    mask = torch.eye(len(adjacent_array)).bool()
    adjacent_array[mask] = 0

    edge_begin_end = create_edge_list(adjacent_array)
    num_node = adjacent_array.size(1)
    num_edge = torch.sum(adjacent_array) // 2
    adjacent_num = round(0.05 * num_node)
    child_chromosome = {
        "genome": None,
        "clusters": None,
        "fitness_1": None,
        "fitness_2": None,
    }

    start_time = time.time()
    EP = []
    idealp = -float("inf") * torch.ones(2)
    weights, neighbors = init_weight(pop_size, num_neighbor)
    chromosome = initial_pnm(
        pop_size,
        adjacent_array,
        adjacent_num,
        num_node,
        threshold
    )
    chromosome = evaluate_objectives(
        chromosome,
        pop_size,
        num_node,
        num_edge,
        adjacent_array,
        pre_result
    )
    f = torch.tensor([c.fitness_1 for c in chromosome])
    idealp = torch.min(f, dim=0).values

    for t in range(maxgen):
        for pop_id in range(pop_size):
            selected_neighbor_id = []
            while (
                not selected_neighbor_id
                or selected_neighbor_id[0] == selected_neighbor_id[1]
            ):
                selected_neighbor_id = torch.randint(0, num_neighbor, (2,))

            selected_pop_id = neighbors[pop_id, selected_neighbor_id]
            child_chromosome[pop_id] = crossover_oneway(
                chromosome, selected_pop_id, num_node
            )

            if torch.rand(1) < p_mu_mi:
                child_chromosome[pop_id] = mutation(
                    child_chromosome[pop_id],
                    p_mutation,
                    num_edge,
                    edge_begin_end
                    )
            else:
                child_chromosome[pop_id] = migration(
                    child_chromosome[pop_id],
                    num_node,
                    adjacent_array,
                    p_migration
                )

            child_chromosome[pop_id] = evaluate_objectives(
                child_chromosome[pop_id],
                1,
                num_node,
                num_edge,
                adjacent_array,
                pre_result,
            )

            for k in neighbors[pop_id]:
                child_fit = decomposed_fitness(
                    weights[k],
                    child_chromosome[pop_id].fitness_1,
                    idealp
                )
                gbest_fit = decomposed_fitness(
                    weights[k],
                    chromosome[k].fitness_1,
                    idealp
                )

                if child_fit < gbest_fit:
                    chromosome[k] = child_chromosome[pop_id]

        for pop_id in range(pop_size):
            if not EP:
                EP.append(chromosome[pop_id])
            else:
                isDominate = False
                isExist = False
                rmindex = []

                for k in range(len(EP)):
                    if torch.equal(chromosome[pop_id].clusters, EP[k].clusters):
                        isExist = True

                    if dominate(chromosome[pop_id], EP[k]):
                        rmindex.append(k)
                    elif dominate(EP[k], chromosome[pop_id]):
                        isDominate = True

                EP = [EP[i] for i in range(len(EP)) if i not in rmindex]

                if not isDominate and not isExist:
                    EP.append(chromosome[pop_id])

            idealp = torch.min(
                torch.cat(
                    [
                        child_chromosome[pop_id].fitness_1.unsqueeze(0),
                        idealp.unsqueeze(0),
                    ]
                ),
                dim=0,
            ).values

    Modularity = torch.tensor([abs(front.fitness_2[0]) for front in EP])
    index = torch.argmax(Modularity).item()
    Division = EP[index].clusters
    Mod = -EP[index].fitness_2[0]
    time_used = time.time() - start_time

    return Mod, chromosome, Division, time_used


def initial_pnm(pop_size, adjacent_array, adjacent_num, num_node, threshold):
    """
    Generate the initial population by PNM

    Args:
        pop_size: the population size
        adjacent_array: the adjacent matrix
        adjacent_num: the number of central nodes in population generation process
        num_node: the number of nodes in the network
        threshold: R=1-threshold is the parameter related to population generation

    Returns:
        chromosome: the generated initial population
    """
    chromosome = [
        {"genome": None, "clusters": None, "fitness_1": 0.0, "fitness_2": 0.0}
        for _ in range(pop_size)
    ]

    A = (adjacent_array > 0).float()
    Doc = A.clone()
    L = adjacent_array.clone()
    T = 1

    for t in range(T):
        Array = Doc / L
        Array[torch.isnan(Array)] = 0

        PRecod = {}
        for i in range(num_node):
            P = solve_physarum2(Array, i)
            Q = caculate_q(P, Array)
            Doc = update_d(Q, Doc)
            PRecod[i] = Q

    Q = average_p(PRecod)
    M = rate_q(Q, threshold)

    for population_id in range(pop_size):
        temp = -A
        adjacent_node = torch.randint(0, num_node, (adjacent_num,))

        for i in range(adjacent_num):
            tempenode = adjacent_node[i]

            for j in range(num_node):
                if A[tempenode, j] == 1 and Q[tempenode, j] < M:
                    temp[tempenode, j] = 1
                    temp[j, tempenode] = 1

        chromosome[population_id]["genome"] = temp

    return chromosome


def average_p(PRecod):
    l = len(PRecod)
    TP = PRecod[0].clone()

    for i in range(1, l):
        TP += PRecod[i]

    AP = TP / l
    return AP


def caculate_q(AP, Array):
    indices = torch.where(Array > 0)
    x, y = indices
    Q = Array.clone()

    Q[x, y] = Array[x, y] * torch.abs(AP[x] - AP[y])

    return Q


def update_d(Q, D):
    u = 1
    r = 0.5
    indices = torch.where(Q > 0)
    x, y = indices

    D[x, y] = r * (Q[x, y] ** u + D[x, y])

    return D


def solve_physarum2(D, outlet):
    n = D.size(0)
    I0 = 10
    Source = torch.zeros(n) + I0
    Source[outlet] = -I0 * (n - 1)

    NewMatrix = D.clone()
    S = torch.sum(D, dim=1)

    indices = torch.arange(n)
    NewMatrix[indices, indices] = -S

    NewMatrix[-1, outlet] = 1000
    Source[-1] = 0
    P = torch.linalg.solve(NewMatrix, Source)

    return P


def rate_q(A, r):
    Z = A[A > 0]
    l = len(Z)
    R = round(l * r)
    Z, _ = torch.sort(Z)
    M = Z[R].item()

    return M


def migration(chromosome, node_num, adj_mat, p_migration):
    """
    Execute migration on a chromosome

    Args:
        chromosome: the chromosome to be migrated
        node_num: the number of nodes
        adj_mat: the adjacent matrix
        p_migration: the migration rate

    Returns:
        chromosome: the migrated chromosome
    """
    # the nodes' communities in a vector
    clu_assignment = change(chromosome, 1, node_num)
    clu_num = torch.max(clu_assignment).item()
    index = [torch.where(clu_assignment == i)[0] for i in range(1, clu_num + 1)]

    for j in range(clu_num):
        num_node_in_clu = len(index[j])
        k = 0

        while k < num_node_in_clu and num_node_in_clu != 0:
            S = adj_mat[index[j]][:, index[j]]

            sum_inter = []
            neighbor_cluster = []

            node_id = index[j][k]
            sum_intra = torch.sum(S[k])
            neighbor_nodes = torch.where(adj_mat[node_id] == 1)[0]
            neighbor_cluster = torch.unique(clu_assignment[neighbor_nodes])
            neighbor_cluster = neighbor_cluster[neighbor_cluster != j + 1]
            len_nc = len(neighbor_cluster)

            if len_nc == 0:
                k += 1
            else:
                sum_inter = torch.zeros((len_nc, 2))
                sum_inter[:, 0] = neighbor_cluster

                for l in range(len_nc):
                    neighbor_clu_id = neighbor_cluster[l]
                    sum_inter[l, 1] = torch.sum(
                        adj_mat[index[neighbor_clu_id - 1]][:, node_id]
                    )

                max_inter = torch.max(sum_inter[:, 1]).item()
                temp_id = torch.where(sum_inter[:, 1] == max_inter)[0]
                max_inter_id = sum_inter[
                    temp_id[torch.randint(len(temp_id), (1,))], 0
                ].item()

                if sum_intra < max_inter:
                    orgn_edge = torch.where(chromosome["genome"][node_id] == 1)[0]
                    chromosome["genome"][orgn_edge, node_id] = -1
                    chromosome["genome"][node_id, orgn_edge] = -1

                    a = torch.where(
                        chromosome["genome"][index[max_inter_id - 1]][:, node_id] == -1
                    )[0]
                    new_edge = index[max_inter_id - 1][a]

                    num_selected_edge = torch.randint(1, len(new_edge) + 1, (1,)).item()
                    selected_edge = new_edge[
                        torch.randperm(len(new_edge))[:num_selected_edge]
                    ]

                    chromosome["genome"][selected_edge, node_id] = 1
                    chromosome["genome"][node_id, selected_edge] = 1

                    clu_assignment[node_id] = max_inter_id
                    index[j] = index[j][index[j] != node_id]
                    index[max_inter_id - 1] = torch.cat(
                        (index[max_inter_id - 1], torch.tensor([node_id]))
                    )
                    num_node_in_clu -= 1

                if sum_intra == max_inter:
                    if torch.rand(1) > p_migration:
                        orgn_edge = torch.where(chromosome["genome"][node_id] == 1)[0]
                        chromosome["genome"][orgn_edge, node_id] = -1
                        chromosome["genome"][node_id, orgn_edge] = -1

                        a = (
                            chromosome["genome"][index[max_inter_id - 1]][:, node_id]
                            == -1
                        )
                        new_edge = index[max_inter_id - 1][a]

                        chromosome["genome"][new_edge, node_id] = 1
                        chromosome["genome"][node_id, new_edge] = 1

                        clu_assignment[node_id] = max_inter_id
                        index[j] = index[j][index[j] != node_id]
                        index[max_inter_id - 1] = torch.cat(
                            (index[max_inter_id - 1], torch.tensor([node_id]))
                        )
                        num_node_in_clu -= 1

                if sum_intra > max_inter:
                    k += 1

    return chromosome


def modularity(adj_mat, clu_assignment):
    """
    Calculate the modularity

    Args:
        adj_mat: the adjacent matrix
        clu_assignment: the cluster assignment

    Returns:
        Q: the modularity
    """
    n = torch.max(clu_assignment).item()
    L = torch.sum(adj_mat) / 2
    Q = 0

    for i in range(1, n + 1):
        index = torch.where(clu_assignment == i)[0]
        S = adj_mat[index][:, index]
        li = torch.sum(S) / 2
        di = torch.sum(adj_mat[index])
        Q += li - (di**2) / (4 * L)

    Q /= L
    return Q


def mutation(child_chromosome, mutation_rate, num_edge, edge_begin_end):
    """
    Execute mutation on a chromosome

    Args:
        child_chromosome: the chromosome to be mutated
        mutation_rate: the mutation rate
        num_edge: the number of edges
        edge_begin_end: the edge list

    Returns:
        child_chromosome: the mutated chromosome
    """
    num_mutation = round(num_edge * mutation_rate)

    for _ in range(num_mutation):
        mutation_edge_id = torch.randint(0, num_edge, (1,)).item()
        begin, end = (
            edge_begin_end[mutation_edge_id, 1],
            edge_begin_end[mutation_edge_id, 2],
        )

        child_chromosome["genome"][begin, end] *= -1
        child_chromosome["genome"][end, begin] *= -1

    return child_chromosome


def modularity(adj_mat, clu_assignment):
    """
    Compute modularity of a partition of a graph.

    Args:
        adj_mat (torch.Tensor): Adjacency matrix of the graph.
        clu_assignment (torch.Tensor): Cluster assignment of each node.

    Returns:
        modularity (float): Modularity value.
    """
    n = clu_assignment.max() + 1
    L = adj_mat.sum() / 2
    Q = 0

    for i in range(n):
        index = torch.where(clu_assignment == i)[0]
        S = adj_mat[index][:, index]
        li = S.sum() / 2
        di = adj_mat[index].sum()
        Q += li - (di**2) / (4 * L)

    return Q / L


def mutation(child_chromosome, mutation_rate, num_edge, edge_begin_end):
    """
    Perform mutation on a chromosome.

    Args:
        child_chromosome (dict): Chromosome to be mutated.
        mutation_rate (float): Mutation rate.
        num_edge (int): Number of edges in the graph.
        edge_begin_end (torch.Tensor): Edge list.

    Returns:
        child_chromosome (dict): Mutated chromosome.
    """
    num_mutation = int(np.ceil(num_edge * mutation_rate))

    for _ in range(num_mutation):
        mutation_edge_id = np.random.randint(num_edge)
        i, j = edge_begin_end[mutation_edge_id, 1:].long()
        child_chromosome["genome"][i, j] *= -1
        child_chromosome["genome"][j, i] *= -1

    return child_chromosome


def calculate_objectives(chromosome, adj_mat, pre_cluster):
    """
    Calculate modularity and normalized mutual information (NMI) of a chromosome.

    Args:
        chromosome (dict): Chromosome to be evaluated.
        adj_mat (torch.Tensor): Adjacency matrix of the graph.
        pre_cluster (torch.Tensor): Previous cluster assignment.

    Returns:
        chromosome (dict): Chromosome with fitness values.
    """
    node_num = adj_mat.shape[0]
    edge_num = adj_mat.sum() / 2

    node_chrom = change(chromosome, 1, node_num)
    clusters_num = node_chrom.max() + 1

    e = torch.zeros(clusters_num)
    a = torch.zeros(clusters_num)

    for j in range(clusters_num):
        nodes_in_cluster = torch.where(node_chrom == j)[0]
        L = len(nodes_in_cluster)

        for k in range(L):
            neighbors = torch.where(adj_mat[nodes_in_cluster[k]] == 1)[0]
            same_cluster = chromosome["genome"][nodes_in_cluster[k], neighbors] == 1
            e[j] += same_cluster.sum()
            a[j] += (~same_cluster).sum()

    e /= 2
    a /= 2
    a += e
    e /= edge_num
    a = (a / edge_num) ** 2

    Q = (e - a).sum()

    chromosome["fitness_1"][0] = -Q
    chromosome["fitness_1"][1] = -nmi(node_chrom, pre_cluster)
    chromosome["clusters"] = node_chrom
    chromosome["fitness_2"][0] = -modularity(adj_mat, node_chrom)

    return chromosome


def evolve(
    W_Cube,
    GT_Cube,
    maxgen,
    pop_size,
    num_neighbor,
    p_mutation,
    p_migration,
    p_mu_mi,
    Threshold,
    num_repeat,
):
    """
    Run the evolutionary algorithm for community detection.

    Args:
        W_Cube (list): List of adjacency matrices at each timestep.
        GT_Cube (list): List of ground truth community assignments at each timestep.
        maxgen (int): Maximum number of generations.
        pop_size (int): Population size.
        num_neighbor (int): Number of neighbors for each subproblem.
        p_mutation (float): Mutation rate.
        p_migration (float): Migration rate.
        p_mu_mi (float): Probability of executing mutation or migration.
        Threshold (float): Threshold for population generation.
        num_repeat (int): Number of repeats.

    Returns:
        dynMod (torch.Tensor): Modularity at each timestep.
        dynNmi (torch.Tensor): NMI at each timestep.
        dynPop (list): Population at each timestep.
        dynTime (torch.Tensor): Running time at each timestep.
        ECD_Result (list): Detected community structure at each timestep.
    """
    num_timestep = len(W_Cube)

    dynMod = torch.zeros(num_timestep, num_repeat)
    dynNmi = torch.zeros(num_timestep, num_repeat)
    dynPop = [None] * num_timestep
    dynTime = torch.zeros(num_timestep, num_repeat)
    ECD_Result = [None] * num_timestep

    for r in range(num_repeat):
        for t in range(num_timestep):
            adj_mat = W_Cube[t]

            if t == 0:
                # Use ECD_1 for the first timestep
                dynMod[t, r], dynPop[t], ECD_Result[t], dynTime[t, r] = ECD_1(
                    torch.from_numpy(adj_mat),
                    maxgen,
                    pop_size,
                    p_mutation,
                    p_migration,
                    p_mu_mi,
                    Threshold,
                )
            else:
                # Use ECD_2 for the following timesteps
                pre_cluster = ECD_Result[t - 1][r]
                dynMod[t, r], dynPop[t], ECD_Result[t], dynTime[t, r] = ECD_2(
                    torch.from_numpy(adj_mat),
                    maxgen,
                    pop_size,
                    p_mutation,
                    p_migration,
                    p_mu_mi,
                    num_neighbor,
                    pre_cluster,
                    Threshold,
                )

            dynNmi[t, r] = nmi(ECD_Result[t][r], GT_Cube[t])

    avg_dynMod = dynMod.mean(dim=1)
    avg_dynNmi = dynNmi.mean(dim=1)

    return dynMod, dynNmi, dynPop, dynTime, ECD_Result



def change(chromosome, population_size, node_num):
    # PyTorch implementation of change function
    node_chrom = torch.zeros(population_size, node_num, dtype=torch.long)

    for population_id in range(population_size):
        flag = torch.zeros(node_num, dtype=torch.bool)
        cluster_id = 1
        node_chrom[population_id, 0] = cluster_id

        for row_id in range(node_num):
            if not flag[row_id]:
                flag[row_id] = True
                node_chrom, flag = row_change(
                    chromosome[population_id]["genome"],
                    node_chrom,
                    flag,
                    population_id,
                    node_num,
                    cluster_id,
                    row_id,
                )
                cluster_id += 1

    return node_chrom


def sort_q(chromosome, whole_size, signal):
    # PyTorch implementation of Sort_Q function
    if signal == 1:
        sorted_indices = torch.argsort(
            torch.tensor([c["fitness_1"] for c in chromosome]), descending=True
        )
    elif signal == 2:
        sorted_indices = torch.argsort(
            torch.tensor([c["fitness_2"] for c in chromosome]), descending=True
        )

    chromosome = [chromosome[i] for i in sorted_indices]

    return chromosome


def init_weight(popsize, niche):
    # PyTorch implementation of init_weight function
    weights = torch.zeros(popsize, 2)
    for i in range(popsize):
        weights[i, 0] = i / (popsize - 1)
        weights[i, 1] = (popsize - i - 1) / (popsize - 1)

    leng = weights.size(0)
    distanceMatrix = torch.zeros(leng, leng)

    for i in range(leng):
        for j in range(i + 1, leng):
            distanceMatrix[i, j] = torch.sum((weights[i] - weights[j]) ** 2)
            distanceMatrix[j, i] = distanceMatrix[i, j]

    _, neighbors = torch.topk(distanceMatrix, niche, dim=1, largest=False)

    return weights, neighbors


def evaluate_objectives(
    chromosome, whole_size, num_node, num_edge, adjacent_array, pre_result
):
    # PyTorch implementation of evaluate_objectives function
    node_chrom = change(chromosome, whole_size, num_node)

    for pop_id in range(whole_size):
        clusters_num = node_chrom[pop_id].max().item()
        e = torch.zeros(clusters_num)
        a = torch.zeros(clusters_num)

        for j in range(clusters_num):
            cluster_id = j + 1
            nodes_in_cluster = torch.where(node_chrom[pop_id] == cluster_id)[0]
            L = len(nodes_in_cluster)

            for k in range(L):
                for m in range(num_node):
                    if adjacent_array[nodes_in_cluster[k], m] == 1:
                        if chromosome[pop_id]["genome"][nodes_in_cluster[k], m] == 1:
                            e[j] += 1
                        else:
                            a[j] += 1

        e /= 2
        a /= 2
        a += e
        e /= num_edge
        a = (a / num_edge) ** 2
        Q = (e - a).sum().item()

        chromosome[pop_id]["fitness_1"] = torch.tensor(
            [-Q, -nmi(node_chrom[pop_id], pre_result)]
        )
        chromosome[pop_id]["clusters"] = node_chrom[pop_id]
        chromosome[pop_id]["fitness_2"] = torch.tensor(
            [-modularity(adjacent_array, chromosome[pop_id]["clusters"])]
        )

    return chromosome


def decomposed_fitness(weight, objectives, idealpoint):
    # PyTorch implementation of decomposedFitness function
    weight[weight == 0] = 1e-5
    part2 = torch.abs(objectives - idealpoint)
    return torch.max(weight * part2)


def dominate(x, y):
    # PyTorch implementation of dominate function
    if isinstance(x, dict):
        x = x["fitness_1"]
    if isinstance(y, dict):
        y = y["fitness_1"]

    return torch.all(x <= y) and torch.any(x < y)


def nmi(A, B):
    # PyTorch implementation of NMI function
    assert len(A) == len(B)
    total = len(A)
    A_ids = torch.unique(A)
    B_ids = torch.unique(B)

    MI = 0
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = torch.where(A == idA)[0]
            idBOccur = torch.where(B == idB)[0]
            idABOccur = np.intersect1d(idAOccur.numpy(), idBOccur.numpy())

            px = len(idAOccur) / total
            py = len(idBOccur) / total
            pxy = len(idABOccur) / total
            MI += pxy * math.log2(pxy / (px * py) + 1e-10)

    Hx = 0
    for idA in A_ids:
        idAOccurCount = len(torch.where(A == idA)[0])
        Hx -= (idAOccurCount / total) * math.log2(idAOccurCount / total + 1e-10)

    Hy = 0
    for idB in B_ids:
        idBOccurCount = len(torch.where(B == idB)[0])
        Hy -= (idBOccurCount / total) * math.log2(idBOccurCount / total + 1e-10)

    return 2 * MI / (Hx + Hy)


def row_change(genome, node_chrom, flag, population_id, node_num, cluster_id, row_id):
    # PyTorch implementation of row_change function
    node_chrom[population_id, row_id] = cluster_id

    for col_id in range(node_num):
        if genome[row_id, col_id] == 1 and not flag[col_id]:
            flag[col_id] = True
            node_chrom, flag = row_change(
                genome, node_chrom, flag, population_id, node_num, cluster_id, col_id
            )

    return node_chrom, flag
