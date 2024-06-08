from mpi4py import MPI


def heapify(arr, n, i):
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2
    if l < n and arr[l] > arr[largest]:
        largest = l
    if r < n and arr[r] > arr[largest]:
        largest = r
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)


def heap_sort_parallel(data):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    chunk_size = len(data) // size
    local_data = data[rank * chunk_size: (rank + 1) * chunk_size]

    n = len(local_data)
    for i in range(n // 2 - 1, -1, -1):
        heapify(local_data, n, i)

    for i in range(n - 1, 0, -1):
        local_data[0], local_data[i] = local_data[i], local_data[0]
        heapify(local_data, i, 0)

    sorted_data = comm.gather(local_data, root=0)

    if rank == 0:
        return sorted_data


if __name__ == "__main__":
    data = [5, 3, 8, 4, 2, 1, 9, 6, 7]
    sorted_data = heap_sort_parallel(data)
    if sorted_data:
        sorted_data = [item for sublist in sorted_data for item in sublist]
        print("Sorted Data:", sorted_data)
