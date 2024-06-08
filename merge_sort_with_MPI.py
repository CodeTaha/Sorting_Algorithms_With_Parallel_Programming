from mpi4py import MPI


def merge_sort_parallel(data):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    def merge(left, right):
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result

    def merge_sort(arr):
        if len(arr) <= 1:
            return arr
        mid = len(arr) // 2
        left = arr[:mid]
        right = arr[mid:]
        return merge(merge_sort(left), merge_sort(right))

    chunk_size = len(data) // size
    local_data = data[rank * chunk_size: (rank + 1) * chunk_size]

    local_sorted = merge_sort(local_data)

    sorted_data = comm.gather(local_sorted, root=0)

    if rank == 0:
        return sorted_data


if __name__ == "__main__":
    data = [5, 3, 8, 4, 2, 1, 9, 6, 7]
    sorted_data = merge_sort_parallel(data)
    if sorted_data:
        sorted_data = [item for sublist in sorted_data for item in sublist]
        print("Sorted Data:", sorted_data)
