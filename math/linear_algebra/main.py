matrix_shape = __import__('7-gettin_cozy').cat_matrices2D

arr1 = [[1, 2], [3, 4]]
arr2 = [[5, 6]]
arr3 = [[7], [8]]
arr4 = matrix_shape(arr1, arr2)
arr5 = matrix_shape(arr1, arr3, axis=1)
print(arr4)
print(arr5)
arr1[0] = [9, 10]
arr1[1].append(5)
print(arr1)
print(arr4)
print(arr5)