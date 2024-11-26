import numpy as np
import sys

if __name__ == "__main__":
    print("Create a numpy array of size 10, filled with zeros")
    arr = np.zeros(10)
    print(arr)

    print("Create a numpy array with values ranging from 10 to 49")
    arr = np.arange(10, 50)
    print(arr)

    print("Create a numpy matrix of 2*2 integers, filled with ones")
    arr = np.ones(shape=(2, 2), dtype=np.int8)
    print(arr)

    print("Create a numpy matrix of 3*2 float numbers, filled with ones.")
    arr = np.ones(shape=(3, 2), dtype=np.float64)
    print(arr)

    print(
        "Given the X numpy array, create a new numpy array with the same shape and type as X, filled with ones."
    )
    x = np.ndarray(shape=(3, 4), dtype=np.int8)
    arr = np.ones(shape=x.shape, dtype=x.dtype)
    print(arr)

    print(
        "Given the X numpy matrix, create a new numpy matrix with the same shape and type as X, filled with zeros."
    )
    x = np.ndarray(shape=(3, 4), dtype=np.int8)
    arr = np.zeros_like(x)
    print(arr)

    print("Create a numpy matrix of 4*4 integers, filled with fives.")
    arr = np.ones(shape=(4, 4), dtype=np.int8) * 5
    print(arr)
    arr = np.array([[5] * 4] * 4, dtype=np.int8)
    print(arr)

    print(
        "Given the X numpy matrix, create a new numpy matrix with the same shape and type as X, filled with sevens."
    )
    x = np.zeros(shape=(2, 3), dtype=np.int8)
    arr = np.ones_like(x) * 7
    print(arr)

    print(
        "Create a 3*3 identity numpy matrix with ones on the diagonal and zeros elsewhere."
    )
    arr = np.identity(3)
    print(arr)

    print("Create a numpy array, filled with 3 random integer values between 1 and 10.")
    arr = np.random.randint(size=3, low=1, high=10)
    print(arr)

    print("Create a 3*3*3 numpy matrix, filled with random float values.")
    arr = np.random.rand(3, 3, 3)
    print(arr)

    print("Given the X python list convert it to an Y numpy array")
    x = [1, 2, 3]
    y = np.array(x)
    print(y)

    print("Given the X numpy array, make a copy and store it on Y.")
    x = np.arange(5)
    y = np.copy(x)
    print(y)

    print("Create a numpy array with numbers from 1 to 10")
    arr = np.arange(1, 11)
    print(arr)

    print("Create a numpy array with the odd numbers between 1 to 10")
    arr = np.arange(1, 11, step=2)
    print(arr)

    print("Create a numpy array with numbers from 1 to 10, in descending order.")
    arr = np.arange(10, 0, step=-1)
    print(arr)

    print("Create a 3*3 numpy matrix, filled with values ranging from 0 to 8")
    arr = np.arange(0, 9, dtype=np.int8).reshape(3, 3)
    print(arr)

    print("Show the memory size of the given Z numpy matrix")
    print(sys.getsizeof(arr))
    print(f"{arr.size * arr.itemsize}")

    print("Given the X numpy array, show it's first element")
    arr = np.arange(10, 20)
    print(arr[0])

    print("Given the X numpy array, show it's last element")
    arr = np.arange(10, 20)
    print(arr[-1])

    print("Given the X numpy array, show it's first three elements")
    arr = np.arange(20, 30)
    print(arr[:3])

    print("Given the X numpy array, show all middle elements")
    x = np.array([1, 2, 3, 4, 5, 6])
    print(x[1:-1])

    print("Given the X numpy array, show the elements in reverse position")
    x = np.array([1, 2, 3, 4, 5, 6])
    print(x[::-1])

    print("Given the X numpy array, show the elements in an odd position")
    x = np.array([1, 2, 3, 4, 5, 6])
    print(x[::2])

    print("Given the X numpy matrix, show the first row elements")
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int8)
    print(x[0])

    print("Given the X numpy matrix, show the last row elements")
    print(x[-1])

    print("Given the X numpy matrix, show the first element on first row")
    print(x[0, 0])

    print("Given the X numpy matrix, show the last element on last row")
    print(x[-1, -1])

    print("Given the X numpy matrix, show the middle row elements")
    print(x[1:-1, :])

    print("Given the X numpy matrix, show the first two elements on the first two rows")
    print(x[:2, :2])

    print("Given the X numpy matrix, show the last two elements on the last two rows")
    print(x[-2:, -2:])

    print("Convert the given integer numpy array to float")
    arr = np.arange(0, 10, dtype=np.int8)
    arr = arr.astype(dtype=np.float16)
    print(arr)

    print("Reverse the given numpy array (first element becomes last)")
    arr = np.array([1, 2, 3, 4, 5, 6, 7])
    print(arr[::-1])

    print("Order (sort) the given numpy array")
    arr = np.array([4, 5, 6, 7, 1, 2, 3])
    arr.sort()
    print(arr)

    print("Given the X numpy array, set the fifth element equal to 1")
    arr = np.array([4, 5, 6, 7, 10, 2, 3])
    arr[4] = 1
    print(arr)

    print("Given the X numpy array, change the 50 with a 40")
    arr = np.array([4, 50, 6, 50, 10, 2, 3])
    arr[arr == 50] = 40

    print("Given the X numpy matrix, change the last row with all 1")
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int8)
    arr[-1] = 1
    print(arr)

    print("Given the X numpy matrix, change the last item on the last row with a 0")
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int8)
    arr[-1, -1] = 0
    print(arr)

    print("Given the X numpy matrix, add 5 to every element")
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int8)
    arr += 5
    print(arr)

    print("Given the X numpy array, make a mask showing negative elements")
    arr = np.array([-1, -2, -3, 3, 4, 5], dtype=np.int8)
    x = arr < 0
    print(x)

    print("Given the X numpy array, get the negative elements")
    arr = np.array([-1, -2, -3, 3, 4, 5], dtype=np.int8)
    mask = arr < 0
    x = arr[mask]
    print(x)

    print("Given the X numpy array, get numbers higher than 5")
    arr = np.array([-1, -2, -3, 3, 6, 8], dtype=np.int8)
    mask = arr > 5
    x = arr[mask]
    print(x)

    print("Given the X numpy array, get numbers higher than the elements mean")
    arr = np.array([-1, -2, -3, 3, 6, 8], dtype=np.int8)
    mask = arr > arr.mean()
    x = arr[mask]
    print(f"mean = {arr.mean():.2f}")
    print(x)

    print("Given the X numpy array, get numbers equal to 2 or 10")
    arr = np.array([-1, 2, -3, 10, 6, 8], dtype=np.int8)
    mask = (arr == 2) | (arr == 10)
    x = arr[mask]
    print(x)

    print("Given the X numpy array, return True if none of its elements is zero")
    arr = np.array([-1, 2, -3, 1, 6, 8], dtype=np.int8)
    print(np.all(arr != 0))

    print("Given the X numpy array, return True if any of its elements is zero")
    arr = np.array([-1, 2, -3, 0, 6, 8], dtype=np.int8)
    print(np.any(arr == 0))

    print("Given the X numpy array, show the sum of its elements")
    arr = np.array([-1, 2, -3, 0, 6, 8], dtype=np.int8)
    print(arr.sum())

    print("Given the X numpy array, show the mean value of its elements")
    arr = np.array([-1, 2, -3, 0, 6, 11], dtype=np.int8)
    print(f"mean = {arr.mean():.2f}")

    print("Given the X numpy matrix, show the sum of its columns")
    arr = np.array([[-1, 2], [-3, 0], [6, 11]], dtype=np.int8)
    x = arr.sum(axis=0)
    print(x)

    print("Given the X numpy matrix, show the mean value of its rows")
    arr = np.array([[-1, 2], [-3, 0], [6, 11]], dtype=np.int8)
    x = arr.mean(axis=1)
    print(x)

    print("Given the X numpy array, show the max value of its elements")
    arr = np.array([2, 30, 12, 23], dtype=np.int8)
    print(arr.max())
