from mpi4py import MPI

def shell_sort_parallel(data):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    chunk_size = len(data) // size
    local_data = data[rank * chunk_size: (rank + 1) * chunk_size]

    n = len(local_data)
    gap = n // 2

    while gap > 0:
        for i in range(gap, n):
            temp = local_data[i]
            j = i
            while j >= gap and local_data[j - gap] > temp:
                local_data[j] = local_data[j - gap]
                j -= gap
            local_data[j] = temp
        gap //= 2

    sorted_data = comm.gather(local_data, root=0)

    if rank == 0:
        return sorted_data

if __name__ == "__main__":
    data = [5, 3, 8, 4, 2, 1, 9, 6, 7]
    sorted_data = shell_sort_parallel(data)
    if sorted_data:
        sorted_data = [item for sublist in sorted_data for item in sublist]
        print("Sorted Data:", sorted_data)
