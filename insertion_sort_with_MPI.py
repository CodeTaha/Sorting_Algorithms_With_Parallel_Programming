from mpi4py import MPI

def insertion_sort_parallel(data):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    chunk_size = len(data) // size
    local_data = data[rank * chunk_size: (rank + 1) * chunk_size]

    for i in range(1, len(local_data)):
        key = local_data[i]
        j = i - 1
        while j >= 0 and key < local_data[j]:
            local_data[j + 1] = local_data[j]
            j -= 1
        local_data[j + 1] = key

    sorted_data = comm.gather(local_data, root=0)

    if rank == 0:
        return sorted_data

if __name__ == "__main__":
    data = [5, 3, 8, 4, 2, 1, 9, 6, 7]
    sorted_data = insertion_sort_parallel(data)
    if sorted_data:
        sorted_data = [item for sublist in sorted_data for item in sublist]
        print("Sorted Data:", sorted_data)
