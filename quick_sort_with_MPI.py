from mpi4py import MPI


def quick_sort_parallel(data):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    def quick_sort(arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quick_sort(left) + middle + quick_sort(right)

    chunk_size = len(data) // size
    local_data = data[rank * chunk_size: (rank + 1) * chunk_size]

    local_sorted = quick_sort(local_data)

    sorted_data = comm.gather(local_sorted, root=0)

    if rank == 0:
        return sorted_data


if __name__ == "__main__":
    data = [5, 3, 8, 4, 2, 1, 9, 6, 7]
    sorted_data = quick_sort_parallel(data)
    if sorted_data:
        sorted_data = [item for sublist in sorted_data for item in sublist]
        print("Sorted Data:", sorted_data)
