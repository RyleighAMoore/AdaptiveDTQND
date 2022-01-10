def mergeSort(myList):
    if len(myList) > 1:
        mid = len(myList) // 2
        left = myList[:mid]
        right = myList[mid:]
        # Recursive call on each half
        mergeSort(left)
        mergeSort(right)
        # Iterators for traversing
        i= 0; j=0; k=0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
              myList[k] = left[i]
              i += 1
            else:
                myList[k] = right[j]
                j += 1
            k += 1
        while i < len(left):
            myList[k] = left[i]
            i += 1; k += 1
        while j < len(right):
            myList[k]=right[j]
            j += 1; k += 1

# arr = [11,3,2,4,-2]
# mergeSort(arr)
# print(arr)


def quickSort(arr):
    numElements = len(arr)
    #Base case
    if numElements < 2:
        return arr
    current_position = 0 #Position of the partitioning element
    for i in range(1, numElements): #Partitioning loop
         if arr[i] <= arr[0]:
              current_position += 1
              temp = arr[i]
              arr[i] = arr[current_position]
              arr[current_position] = temp

    temp = arr[0]
    arr[0] = arr[current_position]
    arr[current_position] = temp #Brings pivot to it's appropriate position

    left = quickSort(arr[0:current_position]) #Sorts the elements to the left of pivot
    right = quickSort(arr[current_position+1:numElements]) #sorts the elements to the right of pivot

    arr = left + [arr[current_position]] + right #Merging everything together
    return arr



array_to_be_sorted = [11,3,2,4,-2, 0,0,10]
print("Original Array: ",array_to_be_sorted)
print("Sorted Array: ", quickSort(array_to_be_sorted))


# arr = [11,3,2,4,-2]
# sortedArr = quickSort(arr)
# print(sortedArr)


class Node:
	def __init__(self, key):
		self.left = None
		self.right = None
		self.val = key

def printInorder(root):
	if root:
		printInorder(root.left)
		print(root.val),
		printInorder(root.right)

def printPostorder(root):
	if root:
		printPostorder(root.left)
		printPostorder(root.right)
		print(root.val),

def printPreorder(root):
	if root:
		print(root.val),
		printPreorder(root.left)
		printPreorder(root.right)
