from mpi4py import MPI
import numpy as np

def insertion_sort(bucket):
    for i in range(1, len(bucket)):
        key = bucket[i]
        j = i - 1
        while j >= 0 and key < bucket[j]:
            bucket[j + 1] = bucket[j]
            j -= 1
        bucket[j + 1] = key

def bucket_sort_parallel(data):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Broadcast the size of the data to all processes
    data_size = len(data) if rank == 0 else None
    data_size = comm.bcast(data_size, root=0)

    # Scatter the data to all processes
    local_data = np.zeros(data_size // size, dtype=np.float64)
    comm.Scatter(data, local_data, root=0)

    max_value = np.max(data) if rank == 0 else None
    max_value = comm.bcast(max_value, root=0)
    size_per_bucket = max_value / size

    # Initialize buckets
    local_buckets = [[] for _ in range(size)]

    # Fill the buckets
    for val in local_data:
        index = int(val / size_per_bucket)
        if index != size:
            local_buckets[index].append(val)
        else:
            local_buckets[size - 1].append(val)

    # Sort each local bucket
    for bucket in local_buckets:
        insertion_sort(bucket)

    # Gather all sorted buckets at root
    sorted_data = comm.gather(local_buckets, root=0)

    if rank == 0:
        # Flatten the list of buckets
        result = []
        for sublist in sorted_data:
            for bucket in sublist:
                result.extend(bucket)
        return result

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        data = np.array([0.78, 0.17, 0.39, 0.26, 0.72, 0.94, 0.21, 0.12, 0.23, 0.68], dtype=np.float64)
    else:
        data = None

    sorted_data = bucket_sort_parallel(data)
    if rank == 0:
        print("Unsorted Array:", data)
        print("Sorted Array:", sorted_data)
