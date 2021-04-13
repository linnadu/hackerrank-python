"""
if intervals:
            raise Exception("0")
"""
print("########################### Bubble Sort  ###################")
"""
It is a comparison-based algorithm in which each pair of adjacent elements
is compared and the elements are swapped if they are not in order.

"""
def BubbleSort(arr):
    n = len(arr)
    for i in range(n-1, 0, -1):
        for j in range(i):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr        

def BubbleSort2(arr): 
    for i, num in enumerate(arr): 
        try: 
            if arr[i+1] < num: 
                arr[i] = arr[i+1] 
                arr[i+1] = num 
                bubble_sort(arr) 
        except IndexError: 
            pass
    return arr 

print("####################### Merge Sort ###########################")
"""
Merge sort first divides the array into equal halves
and then combines themin a sorted manner.
"""
def MergeSort(arr):
    if len(arr) < 2: return arr
    m = len(arr) // 2
    arr1 = MergeSort(arr[:m])
    arr2 = MergeSort(arr[m:])
    l, r = 0, 0
    arr_merge = []
    while l < len(arr1) and r < len(arr2):
        if arr1[l] < arr2[r]:
            arr_merge.append(arr1[l])
            l += 1
        else:
            arr_merge.append(arr2[r])
            r += 1
    if arr1[l:]: arr_merge += arr1[l:]
    if arr2[r:]: arr_merge += arr2[r:]          
    return arr_merge

print("######################Quick Sort#################################")
print("Pivot the last")
def partition(arr, l, r):
    i = l - 1
    p = arr[r]
    print("i", i, p)
    for j in range(l, r):
        if arr[j] < p:
            i += 1
            print(i,j)
            print(arr)
            arr[i], arr[j] = arr[j], arr[i]
            print (arr)
    print(arr)
    arr[i+1], arr[r] = arr[r], arr[i+1]
    print(arr)
    print(i+1)
    return i + 1
def QuickSort(arr, l, r):
    if l < r:
        p = partition(arr, l, r)
        print("p", p)
        QuickSort(arr, l, p - 1)
        QuickSort(arr, p+1, r)
    return arr

def QuickSortR(arr, l, r):
    if l < r:
        p = arr[r]
        i = l - 1
        for j in range(l, r):
            if arr[j] < p:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[r] = arr[r], arr[ i+1 ]
        QuickSortR(arr, l, i )
        QuickSortR(arr, i + 2, r)
    return arr

print("Pivot the first")
def partitionL(arr, l, r):
    p = arr[l]
    i = r + 1
    for j in range(r, l, -1):
        if arr[j] > p:
            i -= 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i - 1], arr[l] = arr[l], arr[i - 1 ]
    return i - 1

def QuickSortL(arr, l, r):
    if l < r:
        p = partitionL(arr,l,r)
        QuickSortL(arr, l, p - 1)
        QuickSortL(arr, p+1, r)
    return arr


print("Pivot the random")
"""
swap the random pivot to the last, than use the last algo.
"""

import random
def partitionRan(arr, l, r):
    k = random.randint(l,r)
    arr[k], arr[r] = arr[r], arr[k]
    p = arr[r]
    i = l - 1
    for j in range(l, r):
        if arr[j] < p:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[r] = arr[r], arr[i + 1]
    return i + 1       
def QuickSortRan(arr, l, r):
    if l < r:
        p = partitionRan(arr, l, r)
        QuickSortRan(arr, l, p - 1)
        QuickSortRan(arr, p + 1, r)
    return arr
        
print("######################Selection Sort#################################")
"""
In every iteration of selection sort, the minimum element
(considering ascending order) from the unsorted subarray
is picked and moved to the sorted subarray
"""

def SelectionSort(arr):
    for i in range(len(arr)):
        min_i = i
        for j in range(i + 1, len(arr)):
            if arr[min_i] > arr[j] :
                min_i = j
        arr[min_i], arr[i] = arr[i], arr[min_i]
    return arr


print("######################Insertion Sort#################################")
"""
"""
def InsertionSort(arr): 
    for i in range(1, len(arr)): 
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j] :
            arr[j + 1] = arr[j]
            j-= 1        
        arr[j + 1] = key
    return arr

print("######################Heep Sort#################################")
"""
https://www.programiz.com/dsa/heap-sort

"""

def heapify(arr, n, root): 
    largest = root 
    l = 2 * root + 1	 
    r = 2 * root + 2	 
    if l < n and arr[largest] < arr[l]: 
        largest = l 
    if r < n and arr[largest] < arr[r]: 
        largest = r
    if largest != root:       
       arr[root],arr[largest] = arr[largest],arr[root] # swap
       heapify(arr, n, largest)

def HeapSort(arr):
    n = len(arr) 
    for i in range(n//2, -1, -1):
        heapify(arr, n, i)
    for i in range(n-1, 0, -1): 
        arr[i], arr[0] = arr[0], arr[i] 
        heapify(arr, i, 0)
    return arr

print("Sort Array by Parity II, even/odd")
def sortArrayByparityII(arr):
    e,o = 0, 1
    while e < len(arr):
        if arr[e] % 2:
            while arr[o] % 2:
                o += 2
            arr[e], arr[o] = arr[o], arr[e]
        e += 2
    return arr


print("Largest Perimeter Trangle")
def largestPerimeter(arr):
        arr.sort()
        for i in range( len(arr), 2, -1):
            if arr[i-3] + arr[i-2] > arr[i-1]:
                return arr[i-1] + arr[i-2] + arr[i - 3]
        return 0

print("Meeting Rooms")
def canAttendMeeting(intervals):
    intervals.sort()
    for i in range(0, len(intervals) - 1):
        if intervals[i][1]> intervals[i+1][0]:
            return False
    return True
        
print("Intersection of Two arrays with duplicate")
def intersect(nums1, nums2):
    dict_nums1 = Counter(nums1)
    output = []
    for i in nums2:
        if i in dict_nums1 and dict_nums1[i]>0:
            output.append(i)
            dict_nums1 -= 1
    return output
            

print("Intersection of Two arrays without duplicate")
def intersectNoDuplicate(nums1, nums2):
    dict_nums1 = Counter(nums1)
    output = []
    for i in nums2:
        if i in dict_nums1 and dict_nums1[i]>0:
            output.append(i)
            dict_nums1 -= 0
    return output

print("K Closest Points to Origin")
def KClosetPoints(points, k):
    cal = lambda x: x[0]**2 + x[1]**2
    def partition(arr, l, r):
        p = arr[r]
        i = l - 1
        for j in range(l, r):
            if cal(arr[j]) < cal(p):
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i+1], arr[r] = arr[r], arr[i+1]
        return i + 1
        
    def sortK(arr, l, r, k):
        if l < r:
            p = partition(arr, l, r)
            if p == k: return
            elif p < k:
                sortK(arr, p + 1, r, k)
            else:
                sortK(arr, l, p - 1, k)   
    sortK(points, 0, len(points) - 1, k)
    return points[:k]
print("sort linked list")
def insertionSortList(self, head):
    cur = dummy = ListNode(0)
    while head:
        if cur and cur.val > head.val: 
            cur = dummy
        while cur.next and cur.next.val < head.val: 
            cur = cur.next
        cur.next, cur.next.next, head = head, cur.next, head.next 
    return dummy.next

print("Reorganize String - no repeat next to each other")
"""A heap is a natural structure to repeatedly
return the current top 2 letters with the largest remaining counts.     
"""

print("high Five, top finve scores in the order of id")
def highFive(items) :
    dic = {}
    for i in items:
        dic.setdefault(i[0], []).append(i[1])
    for i in dic.keys():
        ave = floor(mean(sorted(dic[i], reverse = True)[:5]))
        dic[i] = ave
    return dic.items()
print("robot return to origin")
def judgeCircle(moves):
    # moves is a string of UDLR
    def cal (s):
        if i == "U": return [0, 1]
        if i == "D": return [0, -1]
        if i == "L": return [-1, 0]
        if i == "R": return [1, 0]
    x, y = 0, 0   
    for i in moves:
        x += cal(i)[0]
        y += cal(i)[1]
    if x==0 and y ==0: return True
    else: return False
                




print("######################Test Cases ################################")

n, m = 1000, 6

b = BubbleSort(random.sample(range(n), m))
print("BubbleSort:", b == sorted(b))


m2 = MergeSort(random.sample(range(n), m))
print("MergeSort:", m2 == sorted(m2))

#qs = QuickSort(random.sample(range(n), m), 0, m- 1)
#print(qs == sorted(qs))

print("#######################################################")


qr = QuickSortR(random.sample(range(n), m), 0, m- 1)
print("Quick Sort, pivot the last", qr == sorted(qr))


ql = QuickSortL(random.sample(range(n), m), 0, m- 1)
print("Quick Sort, pivot the fist", ql == sorted(ql))

qRan = QuickSortRan(random.sample(range(n), m), 0, m- 1)
print("Quick Sort, pivot the random", qRan == sorted(qRan))

SS = SelectionSort(random.sample(range(n), m))
print("Selection Sort:", SS== sorted(SS))

IS = InsertionSort(random.sample(range(n), m))
print("Insertion Sort:", IS== sorted(IS))

HS = HeapSort(random.sample(range(n), m))
print("Heap Sort:", HS == sorted(HS))

print("######################################################################")

                
                

