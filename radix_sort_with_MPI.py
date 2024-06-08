from mpi4py import MPI

def radix_sort_parallel(data):
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

    def counting_sort(arr, exp):
        output = [0] * len(arr)
        count = [0] * 256
        for i in range(len(arr)):
            index = arr[i] // exp
            count[index % 256] += 1
        for i in range(1, 256):
            count[i] += count[i - 1]
        i = len(arr) - 1
        while i >= 0:
            index = arr[i] // exp
            output[count[index % 256] - 1] = arr[i]
            count[index % 256] -= 1
            i -= 1
        for i in range(len(arr)):
            arr[i] = output[i]

    chunk_size = len(data) // size
    local_data = data[rank * chunk_size: (rank + 1) * chunk_size]

    max_element = max(local_data)
    exp = 1
    while max_element // exp > 0:
        counting_sort(local_data, exp)
        exp *= 256

    sorted_data = comm.gather(local_data, root=0)

    if rank == 0:
        return sorted_data

if __name__ == "__main__":
    data = [5, 3, 8, 4, 2, 1, 9, 6, 7]
    sorted_data = radix_sort_parallel(data)
    if sorted_data:
        sorted_data = [item for sublist in sorted_data for item in sublist]
        print("Sorted Data:", sorted_data)
