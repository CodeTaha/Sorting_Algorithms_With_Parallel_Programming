from mpi4py import MPI

def selection_sort_parallel(data):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    chunk_size = len(data) // size
    local_data = data[rank * chunk_size: (rank + 1) * chunk_size]

    for i in range(len(local_data)):
        min_idx = i
        for j in range(i + 1, len(local_data)):
            if local_data[j] < local_data[min_idx]:
                min_idx = j
        local_data[i], local_data[min_idx] = local_data[min_idx], local_data[i]

    sorted_data = comm.gather(local_data, root=0)

    if rank == 0:
        return sorted_data

if __name__ == "__main__":
    data = [5, 3, 8, 4, 2, 1, 9, 6, 7]
    sorted_data = selection_sort_parallel(data)
    if sorted_data:
        sorted_data = [item for sublist in sorted_data for item in sublist]
        print("Sorted Data:", sorted_data)
