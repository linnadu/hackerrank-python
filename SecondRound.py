# -*- coding: utf-8 -*-
# 387 first unique char in a string
class Solution(object):
    def firstUniqChar(self, s):
        dic = Counter(s)
        for i in range(len(s)):
            if dic[s[i]] == 1:
                return i
        return -1
'''
############################################################################################################################################################
##############################################################################
##############################################################################
################Sliding                            Windows ###################
################Sliding                            Windows ###################
################Sliding                            Windows ###################
################Sliding                            Windows ###################
################Sliding                            Windows ###################
################Sliding                            Windows ###################
##############################################################################
##############################################################################
##############################################################################
'''
#438. Find all anagrams in a string
# s: "cbaebabacd"; p: 'abc'
#:  n: size of s, m: size of p
#     for each i in s: o(n)
#     we look for the substring ending in i with a lengh m i
#     check if the substring, s[i - m: i] is an anagram O(m)
#     --- instead of check each substring, we use a matchcount to check the length o(1)
#          ------dic: {a:1;b:1;c:1} we define a map to represent that we found all char in p, if found a match, dic: {a:0;b:0;c:0}
#          ------matchcount: represent the length of all char are found: matchcount == 3
#                ---- remove the first element of the substring: --map--; mactchcount--
#                -----add the new i to the end of the substring -- map++; matchcount++
                 
class Solution(object):
    def findAnagrams(self, s, p):
        mapP = Counter(p) # datastructure of p, {a:1; b:1; c:1}. we look for {a:0;b:0;c:0}
        matchCount = 0 # length request for anagram
        m = len(p)
        output = []
        for i in range(len(s)):
            if s[i] in mapP:  # add new element
                mapP[s[i]] -= 1
                if mapP[s[i]] >= 0:# count up prevent extra a,b,c than needed
                    matchCount += 1
            if i - m >= 0:  # length of substring greater than m 
                if s[i - m] in mapP:  # delete first element
                    mapP[s[i - m]] += 1
                if mapP[s[i-m]]>= 1:
                    matchCount -= 1
            if matchCount == len(p):
                output.append(i - m + 1)
        return output
        

#340 Longest substring with at most K distinct characters
# s = 'eceba', k = 2: output: 3; s = 'aa', k = 1; output 2
# for each i in s:
  # find k distinct charcters  ---- use map to find 
  # comapring the length of substring --- use a length count to comparing
  #  ---- remove the first element from the substring if the new char not in dic
  #  ----- add new char to substring if new is already in map
class Solution:
    def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
        '''
        for each i in s:
            find all the substring ended with i that has <= k distict
            the leftmost idx j 
        DS: {char: # of char found}  <= k
        algo: traverse + SW
                         SW[leftmost idx, idx]: has at most than k distinct char
                       update the maxlength if it's longer
              Add: when find a distinct char: dic ++, match count ++
              Remove: when match count > k:   dic --, match count --
              update maxlengh: count == k            
        '''
        if not s or k == 0: return 0
        dic = {}  
        matchcount = 0
        maxLength = 0 # length 
        j = 0
        for i in range(len(s)):
            if s[i] not in dic:
                dic[s[i]] = 1
            else:
                dic[s[i]] += 1
            while j <= i and len(dic) > k:
                dic[s[j]] -= 1
                if dic[s[j]] == 0:
                    del dic[s[j]] 
                j += 1
            maxLength = max(maxLength, i - j + 1) 
        return maxLength 


#76. Minimum Window Substring
'''
data structure S:
	map<char, int: # of chars need to be matched>
	matchCount: # of chars that are already matched

add: right pointer to the right of the window
for each i in s:
        found a char in t, # of char  -- 
		if tMap[s[i]] 1→ 0: matchCount++;
check 当[slow, fast] 满足的时候，做什么： ⇒ update globalMin = min(globalMin, r-l+1)
	matchCount == tMap.size():
remove:  slow from left to right, remove s[slow]  from left of the window
                   found a new char also in t that sate, 
	       move l++ until l is the rightMost position, where [l ,r] contains all chars in t
	     如果window内物理意义被打破:  matchCount != tMap.size()
		check matchCount: if tMap[s[l]] 0→ 1  matchCount++;

'''
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        # look for {a:1; b:1;c:1}
        # substring has this s[i: j] that contains the target
        # min
        # i - j + 1 to measure the length of the head
        # match count and dic to masurea the target
        dic = Counter(t)
        matchcount = 0
        j = 0 
        output = ''
        for k, v in enumerate(s):
            if v in dic:              
                dic[v] -= 1
                if dic[v] >= 0:
                    matchcount += 1
            while matchcount == len(t): # look for rightmost
                if not output or len(output) > k - j + 1:
                    output = s[j: k + 1]
                if s[j] in dic:          # move the right side idx
                    dic[s[j]] += 1  
                if dic[s[j]] > 0:        
                    matchcount -= 1
                j += 1                    
        return output

# 3. Longest Substring Without Repeating Characters
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        #s 
        # for i in s:
        #     substring s[j:i] has no reapeaat
        #hasmap {char: #}
        # {char: 1} 
        # sw: [l, r], r is the current idx
        #             l is the left most idx that has no dup
        #             find dup, move l,
                      # update maxlength
        # add: new, not in c
        # remove: # > 1
        # dic {a:1, b:1}
        #s = "abcbbabcbb"
        # j rightmost
        
        dic = {}
        j = 0
        maxLength = 0
        for i in range(len(s)):   
            if s[i] not in dic:
                dic[s[i]] = i
            else:
                j = max(j, dic[s[i]] + 1) # dic[s[i]] may be outside of j
                dic[s[i]] = i
            maxLength = max(maxLength, i - j + 1)
        return maxLength

# 209 Minimum Size Subarray Sum
#Given an array of positive integers nums and a positive integer target,
#return the minimal length of a contiguous subarray [numsl, numsl+1, ..., numsr-1, numsr]
#of which the sum is greater than or equal to target. If there is no such subarray, return 0 instead.
#Input: target = 7, nums = [2,3,1,2,4,3]
#Output: 2
#Brute Force
class Solution:
    def minSubArrayLen(self, target, nums):      
        minCount = inf
        for i in range(len(nums)): 
            for j in range(i + 1):
                sums = 0                
                for k in range(i - j + 1): 
                    sums += nums[i - k]
                if sums >= target:
                    minCount = min(minCount, i - j + 1)
        if minCount != inf:
            return minCount
        else: return 0
        
class Solution:
    def minSubArrayLen(self, target, nums):      
        '''
        for i in each nums:
               find the substring ending with i that sum up greater than target
        DS: queue: leftside pop the leftmost idx, right right append current idx
        algo: traverse + SW
                         [j, idx]: 
                               idx is the curent i
                          j is the leftmost idx
        Add: when the sum is < target
        remove: when the sum is >= target
        update minCount         
        '''
        q = collections.deque()  
        sumN = 0
        minCount = inf
        for i in range(len(nums)):   
            q.append(i)
            sumN += nums[i]
            while sumN >= target:
                j = q.popleft()
                minCount = min(minCount, i - j + 1)
                sumN -= nums[j]
        if minCount != inf:
            return minCount
        else:
            return 0


#713. Subarray Product Less Than K
#Count and print the number of (contiguous) subarrays
#where the product of all the elements in the subarray is less than k.
#Input: nums = [10, 5, 2, 6], k = 100
#Output: 8
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        '''
        for each i in nums:
            find all substrings ended in i that product to k
            count ++
        DS : queue: idx decreasing queue, left: leftmost, right: idx
        algo: traverse + SW
                         [j, idx]
                             idx: current i
                          any idx that product less than k
        Add:  when product < k
        remove: when product >= k
        
        '''        
        count = 0
        sumP = 1
        if k <=1 : return 0
        j = 0
        for i in range(len(nums)):
            sumP *= nums[i]
            while sumP >= k:
                sumP /= nums[j] 
                j += 1
            count += i - j + 1
        return count


# 239. Sliding Window Maximum
#You are given an array of integers nums, there is a sliding window of size k which is moving from the very left of the array to the very right.
#You can only see the k numbers in the window. Each time the sliding window moves right by one position.
#Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
#Output: [3,3,5,5,6,7]

class Solution(object):
    def maxSlidingWindow(self, nums, k):
        # deck: idx of the max: decreasing queue, with length <= k
        # brudforsce
        # for each i is nums :
        #    max of the substring ended with i; start with i - 3, 
        #    leftmost of deck  is always the max of the SW
        # DS : queue : save the max value for each time move idx, decreasnig queue, 
                     # max, second max, thrid max, forth max
        # when new max comes in, move out the rest, and move it to the leftmost
        # algo: traverse + sliding window 
        #                   [left, idx]
        #                 left:   idx of last max
        # add: larger than the current large
        #       the leftmost value of the queue is always the max 
        # remove: out of k 
        q = collections.deque()
        maxSw = []
        for i in range(len(nums)):                      
            # add the largest, remove the smaller first 
            while q and nums[i]> nums[q[-1]]:
                q.pop()
            q.append(i) 
            if i - q[0] >= k:       ## popleft the one out of SW
                    q.popleft()
            if i + 1 >= k : # length greater than k has a max

                maxSw.append(nums[q[0]])
        return maxSw

#1477. Find Two Non-overlapping Sub-arrays Each With Target Sum
#Given an array of integers arr and an integer target.
#You have to find two non-overlapping sub-arrays of arr each with sum equal target.
class Solution:
    def minSumOfLengths(self, arr: List[int], target: int) -> int:
        ''' 
        for each i in arr:
             find subarry that ended with i sum to target
             rightmost subarry start with j that has non - overlapping
                two subarray
        DS: sumN
        Algo:  sliding window  [j, idx]
        Add: if sumN < target
        Remove if sumN > target
        check  sumN == target: 
                   update minlength
                   update minSum
                   check no overlapping                   
        '''
        minSum = inf
        minlength = [inf]*len(arr) # minlength that ending j 
        j = 0
        sumN = 0
        for i in range(len(arr)):
            sumN += arr[i]
            while sumN > target:
                sumN -= arr[j]
                j += 1
            if sumN == target:
                minSum = min(minSum, minlength[j - 1] + i - j + 1) # j - 1 no-overlapping
                minlength[i] = min(minlength[i - 1], i - j + 1) # record the min
            else:
                minlength[i] = minlength[i - 1]
        return minSum if minSum != inf else -1

#295. Find Median from Data Stream

class MedianFinder:
    def __init__(self):
        self.small = [] 
        self.large = [] 
    def addNum(self, num):
        if len(self.small) == 0:
            heapq.heappush(self.small, -num)
            return
        if num <= -self.small[0]:
            heapq.heappush(self.small, -num)
        else:
            heapq.heappush(self.large, num)
        if len(self.small) - len(self.large) == 2:
            heapq.heappush(self.large, -heapq.heappop(self.small))
        elif len(self.small) - len(self.large) == -2:
            heapq.heappush(self.small, -heapq.heappop(self.large))

    def findMedian(self):
        if len(self.small) == len(self.large):
            return (self.large[0] - self.small[0])/2.0
        return -float(self.small[0]) if len(self.small) > len(self.large) else float(self.large[0])





'''
############################################################################################################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
'''

# 34. find first and last position of element in a array
#Input: nums = [5,7,7,8,8,10], target = 8
#Output: [3,4]
class Solution:
    def searchRange(self, nums, target):
        
        def findLeft(nums, target):
            index = -1
            l, r = 0, len(nums) - 1
            while l <= r:
                mid = (l + r) // 2
                if nums[mid] == target:
                    index = mid 
                    r = mid - 1
                elif nums[mid] > target:
                    r = mid - 1
                else:
                    l = mid + 1
            return index

        def findRight(nums, target):
            l, r = 0, len(nums) - 1
            index = -1
            while l <= r:
                mid = (l + r) // 2
                if nums[mid] == target:
                    index = mid
                    l = mid + 1
                elif nums[mid] > target:
                    r = mid - 1
                else:
                    l = mid + 1
            return index
        return [findLeft(nums, target), findRight(nums,target)]


# 33. Search in Rotated Sorted Array
#Input: nums = [4,5,6,7,0,1,2], target = 0
#Output: 4
#Input: nums = [4,5,6,7,0,1,2], target = 3
#Output: -1
# one pass - revised binary search to narrow down the scop of the search 
#At each iteration, we reduce the search scope into half,
#by moving either the start or end pointer to the middleof the previous search scope.
#
class Solution:
    def search(self, nums, target):
        l, r = 0, len(nums) - 1
        while l <= r:
            m = (l + r) // 2
            if nums[m] == target:
                return m
            elif nums[m] > target:
                if target <= nums[r] and nums[r] < nums[m]: # target is in the roated zone not current zone.
                    l = m + 1 
                else:
                    r = m - 1 # if located in not in the rotated zone, 
            else:
                if target >= nums[l]  and nums[l] > nums[m]:
                    r = m - 1
                else:
                    l = m + 1
        return -1




#875 Koko Eating bananas
#Input: piles = [3,6,7,11], H = 8
#Output: 4
# method: simulate the hours needed with each K, if sumHour < H, it's posssible, otherwise, not possible
# binary search the smallest K that method is True
# we can deduce that Koko finishes it in Math hours
class Solution:
    def minEatingSpeed(self, piles, H):
        if not piles or not H: return 0
        def possible(K):
            summ = 0
            for i in piles:
                if i%K == 0:
                    summ += i//K
                else:
                    summ += i//K + 1
            return summ <= H
               
        l , r = 1, max(piles)
        
        while l <= r:
            mid = (l + r) // 2
            if possible(mid):
                r = mid - 1
            else:
                l = mid + 1
        return l
#1283 Find the Smallest Divisor Given a Threshold
class Solution:
    def smallestDivisor(self, nums, threshold):
        def condition(divisor):
            return sum([(num - 1) // divisor + 1 for num in nums]) <= threshold

        left, right = 1, max(nums)
        while left < right:
            mid = left + (right - left) // 2
            if condition(mid):
                right = mid
            else:
                left = mid + 1
        return left    
#69 Sqrt(x)
class Solution:
    def mySqrt(self, x):
        if x < 2: return x
        l, r = 2, x//2
        while l <= r:
            m = (l + r)//2
            if m*m > x:
                r =  m - 1
            elif m*m < x:
                l = m + 1
            else:
                return m
        return r

# 75. Sort Colors


###########################################

#Happy Number
# log(N)
class Solution:
    def isHappy(self, n):
        if n <= 0: return False 
        def loop(n):
            suma = 0
            while n:
                suma += ( n %10)**2
                n //= 10
            return suma
        visited = set()
        while n != 1 and n not in visited:
            visited.add(n)
            n = loop(n)            
        return n == 1
    
    
#74 search 2D metrix
class Solution(object):
    def searchMatrix(self, matrix, target):
        if not matrix : return False
        m, n = len(matrix), len(matrix[0])
        r, c = 0, n - 1
        while r < m and c >= 0:
            if matrix[r][c] == target:
                return True
            elif matrix[r][c] < target:
                r += 1
            else:
                c -= 1
        return False

#162  Find peak element
# find any peak
#O(logN)
class Solution:
    def findPeakElement(self, nums):
        l, r = 0, len(nums) - 1
        while l < r:
            mid = (l + r) // 2
            if nums[mid] > nums[mid + 1]:
                r = mid 
            else:
                l = mid + 1
        return l



#######################
'''
pure recursion:
https://leetcode.com/problems/same-tree/
invert binary tree
symmetric tree
merge two binary tree
maximum depth of binary tree
minimum depth of binary tree

DFS backtracking
binary tree paths
path sum
path sum II
path sum III
sum root to leaf numbers 
'''

############################################
# BFS
'''
-- explore nodes in “layers”
-- can compute shortest paths -- connected components of undirected graph
-- mark s as explored
-- let Q = queue data structure (FIFO), ini*alized with s
   while Q:
       remove the first node of Q, call it v
       for each edge(v,w) :
           if w unexplored
              mark w as explored
              add w to Q (at the end)
TC: O(m + n): m: # of reachable node; # of reachable edge
Applcation: shortest paths
Goal : compute dist(v), the fewest # of edges on a path from s to v.

'''
# 1. binary search tree: 297, 102, 103, 107, 513, lint-242 - convert Binary tree to linked lists by depth 
# topoligy : lint 127 ( topological sorting);  207; 210. 269
# matrix: 200, 490, 505, 542 733, 994, 305, 773,  lint - 573, 598, 611, 794
# graph: 133, 127, 261, 841, 323, 1306, 531, 618, 624

# tree:
#      94, 144, 145,  105, 106, 889
#      173, 230, 285, 270272, 510 915
#      98, 100, 101, 110
#      111, 104, 333, lint 596, lint 597
# path: 112, 113, 124, lint 475, 298 549 lint - 619
'''
# 112 path Sum
return true if the tree has a root-to-leaf path such that
adding up all the values along the path equals targetSum.
'''
class Solution:
    def hasPathSum(self, root: TreeNode, targetSum: int) -> bool:
        def dfs(root, targetSum):
            if not root: return False
            if not root.left and not root.right and targetSum == root.val: 
                return True
            targetSum -= root.val 
            return dfs(root.left, targetSum ) or dfs(root.right, targetSum )
        return dfs(root, targetSum)


'''
113 path sum ii
Given the root of a binary tree and an integer targetSum,
return all root-to-leaf paths where each path's sum equals targetSum.
'''
class Solution:
    def pathSum(self, root: TreeNode, targetSum: int) -> List[List[int]]:
        output = []
        
        def dfs(root, targetSum, path):
            if not root: return 
            if not root.left and not root.right and targetSum == root.val:
                output.append(path + [root.val])
            return (dfs(root.left, targetSum - root.val, path + [root.val]) 
                    or dfs(root.right, targetSum - root.val, path + [root.val]))        
        dfs(root, targetSum, [])
        return output

longestUnivaluepath()
class Solution(object):
    def longestUnivaluePath(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        # Time: O(n)
        # Space: O(n)
        longest = [0]
        def traverse(node):
            if not node:
                return 0
            left_len, right_len = traverse(node.left), traverse(node.right)
            left = (left_len + 1) if node.left and node.left.val == node.val else 0
            right = (right_len + 1) if node.right and node.right.val == node.val else 0
            longest[0] = max(longest[0], left + right)
            return max(left, right)
        traverse(root)
        return longest[0]



# LCA:  236, lint 474, lint 578
# others: 199, 513, 331, 449, 114

#################################################################################################
# 100 Same Tree
#level order traversal 
class Solution:
    def isSameTree(self, p, q):
        if not q and not p: return True
        if not q or not p: return False
        if q.val == p.val:
            return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        
# 101 Symmetric Tree
# Given a binary tree, check whether it is a mirror of itself
class Solution:
    def isSymmetric(self, root):
        q = collections.deque()
        q.append(root)
        while q:
            n = len(q)
            arr = []
            for i in range(n):
                node = q.popleft()
                if node:
                    q.append(node.left)
                    q.append(node.right)
                    arr.append(node.val)
                else:
                    arr.append(float('-inf'))
            l, r = 0, len(arr) - 1
            print(arr)
            while l <= r:
                if arr[l] != arr[r]:                    
                    return False
                l += 1
                r -= 1
        return True
class Solution:
    def isSymmetric(self, root):
        def mirror(q, p):
            if not q and not p: return True
            if not q or not p: return False
            if p.val == q.val: 
                return mirror(q.left, p.right) and mirror(q.right, p.left)
        return    mirror(root, root)
    
############################################################
# 102 Binary Tree Level Order Traversal
#Given a binary tree, return the level order traversal of its nodes' values
class Solution:
    def levelOrder(self, root):
        if not root: return []
        q, level, output = collections.deque(), 0, []
        q.append(root)
        while q:
            n = len(q)
            output.append([])
            for _ in range(n):
                node = q.popleft()                
                if node.left: q.append(node.left)
                if node.right:q.append(node.right)
                output[level].append(node.val)
            level += 1
        return output


############################################################
# 103 Binary Tree Zigzag Level order Traversal
class Solution:
    def zigzagLevelOrder(self, root):
        if not root: return []
        q, output, level = collections.deque(), [], 0
        q.append(root)
        while q:
            n = len(q)
            arr = deque()
            for _ in range(n):
                node = q.popleft()
                if node.left: q.append(node.left)
                if node.right: q.append(node.right)
                if level % 2 == 0:
                    arr.append(node.val)
                else:
                    arr.appendleft(node.val)
            output.append(arr) 
            level += 1
        return output

###########################################
# 513 Find Bottom Left Tree Value
#Given the root of a binary tree, return the leftmost value in the last row of the tree.
class Solution:
    def findBottomLeftValue(self, root):
        q = collections.deque()
        output = []
        q.append(root)
        while q:
            n = len(q)
            arr = []
            for _ in range(n):
                node = q.popleft()
                if node.left: q.append(node.left)
                if node.right: q.append(node.right)
                arr.append(node.val)
        return arr[0]
#Breadth First or Level Order Traversal : 1 2 3 4 5
#Please see this post for Breadth First Traversal.
#############################################
#Depth First Traversals:
#(a) Inorder (Left, Root, Right) : 4 2 5 1 3
#(b) Preorder (Root, Left, Right) : 1 2 4 5 3
#(c) Postorder (Left, Right, Root) : 4 5 2 3 1

#94 Binary Tree Inorder Traversal
class Solution:
    def inorderTraversal(self, root):       
        output = []
        def f(root):
            if not root: return []
            f(root.left)
            output.append(root.val)
            f(root.right)
        f(root)
        return output    
#144 Binary Tree Preorder Traversal
class Solution:
    def preorderTraversal(self, root):
        output = []
        def f(root):
            if not root: return []
            output.append(root.val)
            f(root.left)
            f(root.right)
        f(root)
        return output
#145 Binary Tree Postorder Traversal
class Solution:
    def postorderTraversal(self, root):
        output = []
        def f(root):
            if not root: return []
            f(root.left)
            f(root.right)
            output.append(root.val)
        f(root)
        return output
#589 N-ary Tree Preorder Traversal 
#Given an n-ary tree, return the preorder traversal of its nodes' values.
#Nary-Tree input serialization is represented in their level order traversal, each group of children is separated by the null value (See examples).
class Node:
    def __init__(self, val = None, children = None):
        self.val = val
        self.children = children
        
class Solution:
    def preorder(self, root):
        output = []
        def f(root):
            if not root: return []
            output.append(root.val)
            for i in root.children:
                f(i)     
        f(root)
        return output
class Solution:
    def preorder(self, root):
        stack, output = [root], []
        while stack:
            node = stack.pop()
            if node:
                output.append(node.val)
                for i in node.children[::-1]:
                    stack.append(i)
        return output

#429. N-ary Tree Level Order Traversal
#Given an n-ary tree, return the level order traversal of its nodes' values.
#Nary-Tree input serialization is represented in their level order traversal, each group of children is separated by the null value (See examples).    
class Solution:
    def levelOrder(self, root):
        if not root: return []
        q = collections.deque()
        output, level = [], 0
        q.append(root)
        while q:
            n = len(q)
            output.append([])
            for _ in range(n):
                node = q.popleft()
                if node:
                    output[level].append(node.val)
                    for i in node.children:
                        q.append(i)                       
            level += 1
        return output
class Solution:
    def levelOrder(self, root):
        output = []
        def f(root, level):
            if not root: return []
            if len(output) == level:
                output.append([])
            output[level].append(root.val)
            for i in root.children:
                f(i, level + 1)
        f(root, 0)
        return output

#590. N-ary Tree Postorder Traversal
#Input: root = [1,null,3,2,4,null,5,6]
#output: Output: [5,6,3,2,4,1]
# 1. from the most left DEEPEST leafs go up, 
# 2. then deepest leafts of next subtree go up
# 3. last the root -> root

class Solution:
    def postorder(self, root):
        output = []
        def f(root):
            if not root: return []
            for i in root.children:
                f(i)
            output.append(root.val)
        f(root)
        return output

# 126. word ladder II
#A transformation sequence from word beginWord to word endWord using a dictionary wordList is a sequence of words such that:
#Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
#Output: [["hit","hot","dot","dog","cog"],["hit","hot","lot","log","cog"]]


######################################################
# DFS
'''
explore aggressively, only backtrack when necessary.
-- also computes a topological ordering of a directed acyclic graph
-- and strongly connected components of directed graphs
DFS (graph G, start vertex s):
    - mark s as explored
    - for every edge(s, v):
        - if v unexplored
        - DFS(G, v)
O(n + m); n: reachable nodes, m: reachable edges
looks at each node in the connected component of s at most once, each edge at most twice.
topological sort (in/out)
DFS (graph G, start vertex s):
    - mark s as explored
    - for every edge(s, v):
        - if v unexplored
        - DFS(G, v)
    - set f(s) = current_lable
    - current_lable = current_label - 1

backtracking
 -- Find a path to success
 -- find all path to success ( # of path, output path)
 --  find the best path
 This procedure is repeated over and over until you reach a final state.
 If you made a good sequence of choices, your final state is a goal state; if you didn't, it isn't.
 Notice that the algorithm is expressed as a boolean function.
 This is essential to understanding the algorithm.
 --------------------------------------------------
 If solve(n) is true, that means node n is part of a solution
 that is, node n is one of the nodes on a path from the root to some goal node.
 We say that n is solvable. If solve(n) is false, then there is no path that includes n to any goal node.
boolean solve(Node n) {
    if n is a leaf node {
        if the leaf is a goal node, return true
        else return false
    } else {
        for each child c of n {
            if solve(c) succeeds, return true
        }
        return false
    }


##################
'''
# 1combination/permudataion. 39, 40, 46, 47, 77, 78, 90, 17, 22, 51, 254, 301, 491, 37, 52, 93, 131, lint - 10, lint 570, lint- 680
# BST: 113, 257, lint - 246, lint 376, lint - 472
# Graph: 140, 494, 1192, 126, 290, 291
#########################################
'''
Time complexity

Permutations: N!
Combinations: N!/((N-k)!K!)

There are generally three strategies to do it:

Recursion

Backtracking
Backtracking is an algorithm for finding all solutions by exploring all potential candidates. If the solution candidate turns to be not a solution (or at least not the last one), backtracking algorithm discards it by making some changes on the previous step, i.e. backtracks and then try again.


Lexicographic generation based on the mapping between binary bitmasks and the corresponding
permutations / combinations / subsets.
As one would see later, the third method could be a good candidate for the interview because it simplifies the problem to the generation of binary numbers, therefore it is easy to implement and verify that no solution is missing.
Besides, this method has the best time complexity, and as a bonus, it generates lexicographically sorted output for the sorted inputs. 
'''
#combinations
################
#Subsets 1
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:        
        def backtrack(idx = 0, chosen = []): # if the combination is done
            if len(chosen) == k:  
                output.append(chosen[:])
                return
            for i in range(idx, n): # add nums[i] into the current combination
                chosen.append(nums[i])
                # use next integers to complete the combination
                backtrack(i + 1, chosen)
                # backtrack
                chosen.pop()        
        output = []
        n = len(nums)
        for k in range(n + 1):
            backtrack(0, [])
        return output
    
def subsets1(self, nums):
    res = []
    self.dfs(sorted(nums), 0, [], res)
    return res
    
def dfs(self, nums, index, path, res):
    res.append(path)
    for i in xrange(index, len(nums)):
        self.dfs(nums, i+1, path+[nums[i]], res)

#subset II
def subsetsWithDup(self, nums):
    res = []
    nums.sort()
    self.dfs(nums, 0, [], res)
    return res
    
def dfs(self, nums, index, path, res):
    res.append(path)
    for i in xrange(index, len(nums)):
        if i > index and nums[i] == nums[i-1]:
            continue
        self.dfs(nums, i+1, path+[nums[i]], res)

def combine(self, n, k):
    res = []
    self.dfs(xrange(1,n+1), k, 0, [], res)
    return res
    
def dfs(self, nums, k, index, path, res):
    #if k < 0:  #backtracking
        #return 
    if k == 0:
        res.append(path)
        return # backtracking 
    for i in xrange(index, len(nums)):
        self.dfs(nums, k-1, i+1, path+[nums[i]], res)
        
#Permutations 1
class Solution:
    def permute(self, nums):
        res = []
        self.dfs(nums, [], res)
        return res

    def dfs(self, nums, path, res):
        if not nums:
            res.append(path)
            #return # backtracking
        for i in range(len(nums)):
            self.dfs(nums[:i]+nums[i+1:], path+[nums[i]], res)
# permutationsn ii
def permuteUnique(self, nums):
    res, visited = [], [False]*len(nums)
    nums.sort()
    self.dfs(nums, visited, [], res)
    return res
    
def dfs(self, nums, visited, path, res):
    if len(nums) == len(path):
        res.append(path)
        return 
    for i in xrange(len(nums)):
        if not visited[i]: 
            if i>0 and not visited[i-1] and nums[i] == nums[i-1]:  # here should pay attention
                continue
            visited[i] = True
            self.dfs(nums, visited, path+[nums[i]], res)
            visited[i] = False

#Combination Sum
class Solution:
    def combinationSum(self, candidates, target):
        res = []
        def dfs(target, index, path):
            if target < 0:
                return  # backtracking
            if target == 0:
                res.append(path)
                return 
            for i in range(index, len(candidates)):
                dfs(target-candidates[i], i, path+[candidates[i]])        
        dfs(target, 0, [])
        return res
#combination Sum II
def combinationSum2(self, candidates, target):
    res = []
    candidates.sort()
    self.dfs(candidates, target, 0, [], res)
    return res
    
def dfs(self, candidates, target, index, path, res):
    if target < 0:
        return  # backtracking
    if target == 0:
        res.append(path)
        return  # backtracking 
    for i in range(index, len(candidates)):
        if i > index and candidates[i] == candidates[i-1]:
            continue
        self.dfs(candidates, target-candidates[i], i+1, path+[candidates[i]], res)

#216. Combination Sum III
#Find all valid combinations of k numbers that sum up to n such that the following conditions are true:



##
## Graph
#Recall that a graph, G, is a tree iff:
#G is fully connected. In other words, for every pair of nodes in G, there is a path between them. visted nodes = len(arr)
#G contains no cycles. : exactly one path between each pair of nodes in G. visited = set() only one time not second


###################
#
# array / sort
#912, 75, 26, 80, 88, /283, 215, | 347, 349, 350, 845, 42, 43, 969, lint: 31, 625, 143, 461, 544, 

################################
# 912 sorted an Array
class Solution:
    def sortArray(self, nums):
        def partition(arr, l, r):
            p = arr[r]
            i = l - 1
            for j in range(l,r):
                if arr[j] < p:
                    i += 1
                    arr[i], arr[j] = arr[j], arr[i]
            arr[i + 1], arr[r] = arr[r], arr[i + 1]
            return i + 1
        
        def qsort(arr, l, r):
            if l < r: 
                p = partition(arr, l, r)
                qsort(arr, l, p - 1)
                qsort(arr, p + 1, r)
            return arr
        qsort(nums, 0, len(nums) - 1)
        return nums

#75 sort/colors(array)
class Solution:
    def sortColors(self, nums):
        i = 0 # last zero 
        j = 0 # last one  
        k = len(nums) - 1  # first 2
        while j <= k:  
            if nums[j] == 0:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
                j += 1
            elif nums[j] == 2:
                nums[j], nums[k] = nums[k], nums[j]
                k -= 1
            else:
                j += 1
#26 Remove Dupliates from sorted Array
class Solution:
    def removeDuplicates(self, nums):
        if not nums: return 0
        i = 1
        while i < len(nums):
            if nums[i] == nums[i-1]:
                nums.pop(i)
                i -= 1
            i += 1
        return i
        

#80 Remove Duplicates from Sorted Array II
class Solution:
    def removeDuplicates(self, nums):
        if not nums: return 0
        i, count = 1, 1
        while i < len(nums):
            if nums[i] == nums[i - 1]:
                count += 1
                if count > 2:
                    nums.pop(i)
                    i -= 1
            else:
                count = 1
            i += 1
        return len(nums)
# 88 Merge sorted Array
# inplace sorting
#Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
#add from the pack since it's zero
#n1, len(nums1)
#n1 - m : 0 
#n2 : len(nums2)
#n1 = m + n, last position of nums1 
# return nums1 ~ inplace sort nums1
# num1 = [1,2,3,  0 ,0 , 0 ]: 0: largest 
# i: element in nums1, m -> 0, 
# j: elements in nums2, n ->0, 
# p: nums1 n1 - >0
# comparing, nums1[i], nums2[j], put the larger one in to nums1[p]
# e.g. comparing, 3, 6,  since 6> 3, repalce nums1[p] = 0, with 6 

class Solution:
    def merge(self, nums1, m, nums2, n):   
        i = m - 1
        j = n - 1
        p = m + n - 1
        while i >=0 and j >= 0: # as long as one arr is iterated complately 
            if nums1[i] >= nums2[j]:
                nums1[p] = nums1[i]
                i -= 1
            else:
                nums1[p] = nums2[i]
                j -= 1
            p -= 1
        # 1) i = 0; j != 0; 2) j = 0, i !=0, 3) i, j == 0 the loop exist
        # if i = 0 : nums1         
        # in the case, nums1[:m] > nums2[::]
        for i in j + 1:
            nums1[i] = nums2[i]
        return nums1 
        O(m+n)
# 283 Move Zeros
#while maintaining the relative order of the non-zero elements.
# move zero to the back
# can not use two pointers from both side
# i: from 0 -> n
# j: from 0 - > n
class Solution:
    def moveZeroes(self, nums):
        i, j = 0, 0
        while j < len(nums):
            if nums[i] != 0:
                i += 1
            else:
                if nums[j] != 0:
                    nums[i], nums[j] = nums[j] , nums[i]
                    i += 1
            j += 1

#215. Kth Largest Element in an Array
#
class Solution:
    def findKthLargest(self, nums, k):
        def partition(arr, l, r):
            rand = random.randint(l, r)
            p = arr[rand]
            arr[r], arr[rand] = arr[rand], arr[r]
            i = l - 1
            for j in range(l, r):
                if arr[j] > p:
                    i += 1
                    arr[j], arr[i] = arr[i], arr[j]
            arr[i + 1], arr[r] = arr[r], arr[i + 1]
            return i + 1
        def sortK(arr, l, r, k):
            if l < r: 
                p = partition(arr, l, r)
                if p == k: return 
                elif p > k: sortK(arr, l, p - 1, k)
                else: sortK(arr, p + 1, r, k)
            else:
                return
        sortK(nums, 0, len(nums) - 1, k-1)
        return nums[k-1]

#347 top K frequent Elements
class Solution:
    def topKFrequent(self, nums, k):
        # frequency of each number
        #  find the first k by frequency (sort, or heap)
        # since heap in python is minheap, use -v to use as maxheap
        hmap = Counter(nums)
        fre = [(-v, i) for i, v in hmap.items()]
        heapq.heapify(fre)
        output = []
        for i in range(k): # first k
            output.append(heapq.heappop(fre)[1])
        return output

# 349 Intersection of Two Array
class Solution:
    def intersection(self, nums1, nums2):
        if len(nums1) > len(nums2): 
            nums1, nums2 = nums2, nums1
        counter = Counter(nums1)
        output = []
        for i in nums2:
            if i not in output and i in counter:
                output.append(i)
        return output
#350. Intersection of Two Arrays II
class Solution:
    def intersect(self, nums1, nums2):
        output = []
        dic1 = Counter(nums1)
        dic2 = Counter(nums2)
        if len(dic1) > len(dic2): dic1, dic2 = dic2, dic1
        for i in dic1:
            if i in dic1 and i in dic2:
                output += [i]* min(dic1[i], dic2[i])
        return output

#42. Trapping Rain water
# for each i  max bewteen [0, i- 1 ]
# for each i  max bewteen  [i + 1, n]
#how much water it can trap after raining.
class Solution:
    def trap(self, height):
        maxleft, maxright, n = 0, 0, len(height)
        waterlevel = n * [0]
        for i in range(n):
            maxleft = max(height[i], maxleft)
            waterlevel[i] = maxleft
        for i in range(1, n+1):
            maxright = max(height[-i], maxright)
            waterlevel[-i] = min(maxright, waterlevel[-i]) - height[-i]
        return sum(waterlevel)

#11. Container with Most Water
#Contains the most water
class Solution:
    def maxArea(self, height):
        l, r = 0, len(height) - 1
        # calculate all the possible combination with l, r bound
        # compare to pick the max
        maxArea = 0
        while l < r:
            if height[l] < height[r]:
                maxArea = max(maxArea, (r - l) * height[l])
                l += 1
            else:
                maxArea = max(maxArea, (r - l) * height[r])
                r -= 1
        return maxArea    

#43. Multiply Strings
#Input: num1 = "123", num2 = "456"
#Output: "56088"   
class Solution:
    def multiply(self, num1, num2):
        s = ''
        output = []
        ans = 0 
        for l2 in range(1, len(num2) + 1):
            carry = 0 
            s = collections.deque()
            for l1 in range(1, len(num1) + 1):           # 123 * 6
                carry += int(num2[-l2]) * int(num1[-l1])# 6 * 3
                s.appendleft(str(carry % 10)) #s + 8
                carry //= 10 # carry = 1
            if carry != 0:
                s.appendleft(str(carry)) # deque push left                
            output.append(''.join(s)) # add a int lie str '738'
            #[123*6, 123*5, 123*4]
        for i in range(len(output)):                
            ans += int(output[i])* (10**i)
        return str(ans)
    
#845. Longest Mountain in Array
# Input: arr = [2,1,4,7,3,2,5]: Output: 5
# Explanation: The largest mountain is [1,4,7,3,2] which has length 5.   
class Solution:
    def longestMountain(self, A):
        maxlength = 0        
        for i in range(1, len(A) - 1):
            if A[i - 1] < A[i] > A[i + 1]:                
                l = r = i     
                while l > 0 and A[l - 1] < A[l]:
                    l -= 1
                while r < len(A) - 1and A[r] > A[r + 1]:
                    r += 1
                maxlength = max(maxlength, (r - l + 1))        
        return maxlength

# 41. First Missing Positive
#Given an unsorted integer array nums, find the smallest missing positive integer.
#Input: nums = [1,2,0] #Output: 3
#Input: nums = [3,4,-1,1] after cycle sort: [1, -1, 3, 4]
#Output: 2
        #It is based on the idea that array to be sorted can be divided into cycles. 
        #Cycles can be visualized as a graph. 
        #We have n nodes and an edge directed from node i to node j 
        # cycle sort, inplace 
class Solution:
    def firstMissingPositive(self, nums) :
        # smallest missing positive integer
        # cycle sort. if the nums[i] in between (1, len(nums)), cycle sort put the nums[i] - 1 to position i,
        # if pisition i is missing, it will put a wrongly number that is outside of (1, len(nums))
        # then we just need to check, to find the i that is not filled with nums[i] - 1 
        #if the element at i-th index must be present at j-th index in the sorted array.
        # put nums[i] - 1 to the correct place if nums[i] in (1, len(nums))
        # so far, all the integers that could find their own correct place 
        # have been put to the correct place, next thing is to find out the
        # place that occupied wrongly
        ###### j is in (1, n) if j is not in (1, n) skip, i += 1
        i, n = 0, len(nums)        
        while i < n:
            j = nums[i] - 1  
            if 0 <= j < n and  nums[i] != nums[j]: # check if position i = nums[i] - 1
                nums[i], nums[j] = nums[j], nums[i]
            else:
                i += 1  #in place do nothing 
        for i in range(n):
            if i != nums[i] - 1:
                return i + 1
        return n + 1
        return n + 1 # no missing

# heapify nlng(n) 
#heapify is inlace O(n)
class Solution:
    def firstMissingPositive(self, nums):
        heapq.heapify(nums)
        num = 1
        for i in range(len(nums)):
            x = heapq.heappop(nums)
            if (x > 0 and x==num):
                num+=1
        return num

####################################################################
#382 470 202, 4 
#382 Linked List Random Node
#Given a singly linked list, return a random node's value from the linked list.
#Each node must have the same probability of being chosen.    
# O(n), O(N)
class Solution:
    def __init__(self, head):
        self.head = head
        arr = []
        while head:
            arr.append(head.val)
            head = head.next
        self.arr = arr        
    def getRandom(self):
        n = len(self.arr)
        seed = random.randint(1, n)
        return self.arr[seed - 1]
########
# Reserve sampling
#O(N),O(1)
#random sampling over a population of 1. unknown size with 2. constant space
# sample K element from the unknow size population
### to make sure    
#1. Initially, we fill up  R[] with the K heading elements from the pool. each being choose Prob = 1/K
#2. We then iterate through the rest of elements in the pool.
#   For each iteration (k+1,  i): 
#       we put the i in to the R[] with Prob, 1/i 
#       we replace one of the K element with prob 1/i,the rest will stay with prob 1/(i-1) * (i-1)/ i
#       where: 1/(i-1): the prob to stay from last round
#              (i-1)/i the prob to stay from this round
#              this prob is a conditional prob, stay this round is conditional on stay from previous round
#   it is guaranteed that at any moment, for each element scanned so far, it has an equal chance to be selected into the reservoir. 
#https://florian.github.io/reservoir-sampling/
#k: samples needed.
# the target is to make sure that the i value has 1/i probability to keep. all i element has equal prob to be sampled
class Solution:
    def __init__(self, head):
        self.head = head
    def getRandom(self):       
        k = 1 # samples needed
        i = k
        sample = 0
        arr = [0]*k
        q = self.head
        while q:
            u = random.random()
            j = random.randint(1, k) #
            if u < 1 / i:
                arr[k-1] = q.val              
            q = q.next
            i += 1
        return arr[k-1]
# 
###############################################
#470
#1/3 + 1/2 * 

''' (k-1) / K * k/n
(1 - k/(n-1))* (k - 1)/k * k/(n+1)
Given the API rand7() that generates a uniform random integer in the range [1, 7], write a function rand10()
that generates a uniform random integer in the range [1, 10]. You can only call the API rand7(), and
you shouldn't call any other API. Please do not use a language's built-in random API.
Each test case will have one internal argument n, the number of times that your implemented function rand10() will be called while testing.
Note that this is not an argument passed to rand10().

Follow up:

What is the expected value for the number of calls to rand7() function?
Could you minimize the number of calls to rand7()?
'''
# 202 happy number
'''
Write an algorithm to determine if a number n is happy.
A happy number is a number defined by the following process:
Starting with any positive integer, replace the number by the sum of the squares of its digits.
Repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1.
Those numbers for which this process ends in 1 are happy.
Return true if n is a happy number, and false if not.
'''
class Solution:
    def isHappy(self, n):
        if n <= 0: return False 
        def loop(n):
            suma = 0
            while n:
                suma += ( n %10)**2
                n //= 10
            return suma
        
        visited = set()
        while n != 1 and n not in visited:
            visited.add(n)
            n = loop(n)            
        return n == 1
    
# 4. Median of Two sorted Arrays
#binary search
#    al  | ar
#    bl  | b
# condition 1: len(left two parts) = len(right two parts): i + j = (m +n) - (i+j) or + 1
  # m1 + m2 = 
# condition 2: max(left two parts) <= max(right two parts): a[i -1] < b[j] and b[j - 1] < a[i]
class Solution:
    def findMedianSortedArrays(self, nums1, nums2):
        if len(nums1) > len(nums2): nums1, nums2 = nums2, nums1 
        n1, n2 = len(nums1), len(nums2)
        l, r = 0, n1
        while l <= r:
            m1 = (l + r) // 2
            m2 = (n1 + n2 + 1) // 2 - m1
            if m1 != 0 and nums1[m1 - 1] > nums2[m2]:
                r -= 1
            elif m1 != n1 and nums1[m1] < nums2[m2 - 1]:
                l += 1
            else:
                if m1 == 0: left = nums2[m2 - 1]
                elif m2 == 0: left = nums1[m1 - 1]
                else: left = max(nums1[m1 - 1], nums2[m2 - 1])
                if (n1 + n2) % 2 == 1:
                    return left
                if m1 == n1: right = nums2[m2]
                elif m2 == n2: right = nums1[m1]
                else: right = min(nums1[m1], nums2[m2])
                return (left + right) / 2

# max_value
# min_value

####
#84. Largest Rectangle in Histogram
#i: left boundary
#j: right boundary from i
#h: mininum height between (i, j)
# area : (j - i + 1) * h
# max_area: max(max_area, (j - i + 1) * h)
#O(n**2)
class Solution:
    def largestRectangleArea(self, heights):
        max_area = 0
        for i in range(len(heights)): # i: left side of each rectangle
            min_height  = inf
            for j in range(i, len(heights)): # right side of each rectangle
                min_height = min(min_height, heights[k]) 
                max_area = max(max_area, min_height*(j - i + 1)) # comparing the area
        return max_area

# optimal solution
# stack = [-1], store the minumum height of each iteration -1: end
#               increaing stack to store the index of last smallest hight. 
# if heights[i] < heights[stack[-1] : pop() the last higher one (last one of the stack), store the new low
#    right side: (i - 1): i is higher than i - 1, leftside: stack[-1]
# else : we append it to the stack.

class Solution:
    def largestRectangleArea(self, heights):
        max_area = 0
        stack = [-1] # the index of left of the rectangle, -1 to mark the end
        heights.append(0)
        for i in range(len(heights)):
            while heights[i]< heights[stack[-1]]: # found the right boundary when current h smaller than lastone
                h = heights[stack.pop()]        # largest height in the past possible companition of rectangular 
                w = (i - 1) - stack[-1]    
                max_area = max(max_area, h*w)
            stack.append(i) # increasing stack ( increaing height), 
        return max_area
## 792. Number of matching subsequences
# bruce force
# w: each words
# i : each char in each w
# check if all i are in s - > yes count += 1 
# for the check: need to check both char and sequence (frequency)
# we use two pointers
# i : position of char in w
# j: begining of search space in s
# for each w[i], we search it in s[j:]
# if we find it, we update j to new position
# tc: len(all chars in words) ** 2
 
class Solution:
    def numMatchingSubseq(self, S, words):       
        def check(w):
            j = 0
            c = 0
            for i in range(len(w)):
                while j <= len(S) - 1:
                    if S[j] == w[i]:
                        c += 1
                        break
                    j += 1
                j += 1
            return c == len(w)       
        count = 0
        for w in words:
            if check(w):
                print(w)
                count += 1
        return count 
#words_dict[a] = ['a','acd,'ace']
#words_dict[b] = ['bb']
#words_dict[d] = ['d']
# if found a, then look for 'cd'
class Solution:
    def numMatchingSubseq(self, S, words):
        word_dict = defaultdict(list)
        count = 0        
        for w in words:
            word_dict[w[0]].append(w)                    
        for i in S:
            words_expecting_char = word_dict[i]
            word_dict[i] = []
            for w in words_expecting_char:
                if len(w) == 1: 
                    count += 1
                else:
                    word_dict[w[1]].append(w[1:])        
        return count 

"""
DataStructure:
array&Matrix: 442, 48, 54, 73, 289
"""
# 442. Find All Duplicates in an Array
#O(n)
class Solution:
    def findDuplicates(self, nums):
        # use hashmap to record the frequncy
        # search the hashmap and return i when frequncy == 2
        dic = Counter(nums)
        output = []
        for i in dic:
            if dic[i] == 2:
                output.append(i)
        return output
# 48 Rotate Image
class Solution:
    def rotate(self, matrix):
        '''         transpose
                     inverse
                     starting from the diagnose matrix[i, j], where i, i                      
                     transpose matrix[i, i +k] with matrix[i + k, i]
                     let j be i + k
                     reflect:
                     for each row: the j switch with n-1-j
         '''
        n = len(matrix)
        for i in range(n):
            for j in range(i, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]                
        for i in range(n):
            for j in range(n//2):
                matrix[i][j], matrix[i][n-1 - j] = matrix[i][n - 1 - j], matrix[i][j]
class Solution(object):
    def verticalTraversal(self, root):
        g = collections.defaultdict(list) 
        queue = [(root,0)]
        while queue:
            new = []
            d = collections.defaultdict(list)
            for node, s in queue:
                d[s].append(node.val) 
                if node.left:  new += (node.left, s-1), 
                if node.right: new += (node.right,s+1),  
            for i in d: g[i].extend(sorted(d[i]))
            queue = new
        return [g[i] for i in sorted(g)]

'''
----------------------------------------------------------------------------------

linked list: 21, 86, 141, 160, 234, 328,  142, 287, 876 
'''
#21. Merge Two Sorted Lists
# why need dummy head? since the end of the while loop,
#the pointer is at the end of either l1 or l2
# also, don't know l1 head is biger or l2 head is bigger
# dummy
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy = curr = ListNode(0)
        while l1  and l2:  # move the pointer whoever is smaller
            if l1.val > l2.val:
                curr.next = l2
                l2 = l2.next
            else:
                curr.next = l1
                l1 = l1.next
            curr = curr.next
        curr.next = l1 or l2 # who ever is left 
        return dummy.next

# 86 Partition List
#Given the head of a linked list and a value x,
#partition it such that all nodes less than x come before nodes greater than or equal to x.
#Input: head = [1,4,3,2,5,2], x = 3
#Output: [1,2,2,4,3,5]
#before which all the elements would be smaller than x and after which all the elements would be greater or equal to x
# both side, don't have to be sorted.


# 141 Linked list cycle
#Given head, the head of a linked list,
#determine if the linked list has a cycle in it.
#a cycle: exist if one node can be reached again by continuously following the next pointer.
#Input: head = [3,2,0,-4], pos = 1

class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        visited = set()
        node = head
        while node is not None:
            if node in visited:
                return True
            else:
                visited.add(node)
                node = node.next
        return None

# 160
# 234 Palindrome Linked List
#Given a singly linked list, determine if it is a palindrome.
# 328
# -----
#142 Linked List Cycle II
# TC O(n), SC O(n)
class Solution(object):
    def detectCycle(self, head):
        visited = set()
        node = head
        while node is not None:
            if node in visited:
                return node
            else:
                visited.add(node)
                node = node.next
        return None
# TC o(n), o(1) # slow / fast
#1 <= nums[i] <= n otherwise will be list out of boundary
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        # Find the intersection point of the two runners.
        slow = fast = nums[0]
        print(slow, fast)
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            print(slow, fast, nums[fast] , nums[nums[fast]])
            if slow == fast:
                break
        print(slow, fast)
        # Find the "entrance" to the cycle.
        slow = nums[0]
        while slow != fast:
            slow = nums[slow]
            fast = nums[fast]        
        return fast 

#206 reverse Linked list
    
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        prev, curr = None, head 
        while curr:
            temp = curr
            curr = curr.next
            temp.next = prev
            prev = temp
        return prev
    
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        def rev(head):
            if not head or not head.next: return head
            p = rev(head.next)
            head.next.next = head
            head.next = None
            return p
        return rev(head)
    
# 143 reorder list

class Solution:
    def reorderList(self, head: ListNode) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        if not head: return []
        # find the mid
        def mid(head):
            s = f = head
            while f and f.next:
                s = s.next
                f = f.next.next
            return s
        # reverse the last half 
        
        def reverse(head):
            if not head or not head.next: return head
            p = reverse(head.next)
            head.next.next = head
            head.next = None
            return p
        
        mid = mid(head)
        rev = reverse(mid)

        # connect 
        first, second = head, rev
                
        while second.next:  
            first.next, first =second, first.next
            second.next, second = first, second.next

#287 Find the Duplicate Number
#Given an array of integers nums containing n + 1 integers
#where each integer is in the range [1, n] inclusive.  
#First of all, where does the cycle come from? Let's use the function f(x) = nums[x] to construct the sequence: x, nums[x], nums[nums[x]], nums[nums[nums[x]]]


#----------
#876 Middle of the Linked List
#Given a non-empty, singly linked list with head node head,
#return a middle node of linked list.
#When traversing the list with a pointer slow,
#make another pointer fast that traverses twice as fast.
#When fast reaches the end of the list, slow must be in the middle.
# TC O(N), O(1)
class Solution:
    def middleNode(self, head: ListNode) -> ListNode:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow
'''
2, 21, 25, 82, 83, 86, 92, 138, 141, 148, 160, 203, 206, 234, 328, 445, 142, 876
'''

#2. Add Two Numbers
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode: 
        pointer = AddTwohead = ListNode(0)
        carry = 0
        while l1 or l2 or carry:
            if l1:
                carry += l1.val
                l1 = l1.next
            if l2:
                carry += l2.val
                l2 = l2.next
            pointer.next = ListNode(carry % 10)
            carry //= 10
            pointer = pointer.next
        return AddTwohead.next

#25. Reverse Nodes in K - Group
#Given a linked list,
#reverse the nodes of a linked list k at a time and return its modified list.   


'''


hash: 706, 49, 128, 560, 953, 290
heap: 23, 295, 347, 692, 767, 973, 480, 703
stack: 155, 20, 85, 224, 227, 394, 1249
monotonic stack: 300, 84, 239, 1019
trie: 208, 211, 1032
union find: 200, 305, 323
sweep line: 252, 253
binary index tree, segment tree: 327, 715, 315, 493
complex data structure: 146, 460, 211, 380, 528, 588, 981, 1396

'''


print("93. Lowest Common Ancestor of a Binary Tree")
#The lowest common ancestor is defined between two nodes p and q as the lowest node in T
#that has both p and q as descendants (where we allow a node to be a descendant of itself).
#Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
#Output: 3: Explanation: The LCA of nodes 5 and 1 is 3.
class Solution:
    def lowestCommonAncestor(self, root, p, q):
             

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        def f(root, p, q):
            if not root: return None
            if root and (root == p or root == q): return root
            else:
                l = f(root.left, p, q)
                r = f(root.right, p, q)
                if root and l and r: return root
                elif l: return l
                elif r: return r
                else:
                    return None
        return f(root, p, q)


class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        # Stack for tree traversal
        stack = [root]
        # Dictionary for parent pointers
        parent = {root: None}

        # Iterate until we find both the nodes p and q
        while p not in parent or q not in parent:

            node = stack.pop()

            # While traversing the tree, keep saving the parent pointers.
            if node.left:
                parent[node.left] = node
                stack.append(node.left)
            if node.right:
                parent[node.right] = node
                stack.append(node.right)

        # Ancestors set() for node p.
        ancestors = set()

        # Process all ancestors for node p using parent pointers.
        while p:
            ancestors.add(p)
            p = parent[p]

        # The first ancestor of q which appears in
        # p's ancestor set() is their lowest common ancestor.
        while q not in ancestors:
            q = parent[q]
        return q

# 85 Maximal Rectangle
#Given a rows x cols binary matrix filled with 0's and 1's,
#find the largest rectangle containing only 1's and return its area.
class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        if not matrix or not matrix[0]: return 0
        maxSum, n = 0, len(matrix[0])
        height = [0]*(n+ 1)
        for r in matrix:
            for l in range(n):
                if r[l] == '1':
                    height[l] += 1
                else:
                    height[l] = 0
            stack = [-1]
            for l in range(n+1):
                while height[l] < height[stack[-1]]:
                    h = height[stack.pop()]
                    w = l - stack[-1] - 1
                    maxSum = max(maxSum, h*w)
                stack.append(l)
        return maxSum




class Solution:
    @functools.lru_cache(None)
    def encode(self, s: str) -> str:
        if len(s) <= 4:
            return s
        n = len(s)
        for seg in range(n, 1,-1):
            if n % seg == 0:
                if s == s[:n // seg] * seg:
                    return str(seg) + '[' + self.encode(s[:n // seg]) + ']'
        ret = s
        for i in range(1,n):
            new = self.encode(s[:i]) + self.encode(s[i:])
            if len(new) < len(ret):
                ret = new
        return ret



class Solution:
    def encode(self, s: str) -> str:
        '''
         base case: s itself is a reapting string if s in s+s
         s is reapting, substring of s is also repaating
        '''
        output = s # if no encode, return s
        # basecase
        encoded = ''
        idx  = (s + s).find(s, 1)
        if 0 < idx < len(s): # s is preating 
            repeatingpattern = s[:idx]
            freq = len(s)//len(repeatingpattern)
            encoded = str(freq) + '[' + self.encode(repeatingpattern) + ']' # if repeating pattern has reapting substring
        if len(encoded) < len(output): 
            output = encoded
        # traverse to find all basecase
        for i in range(1, len(s)):
            encoded = self.encode(s[:i]) + self.encode(s[i+1:])
            if len(encoded) > len(output): 
                output = encoded
        return output

'''
Streaming
'''
# moving average
class MovingAverage:

    def __init__(self, size: int):
        """
        Initialize your data structure here.
        """
        self.Wsize = size # size of the window
        self.queue = deque()
        self.Wsum = 0
        self.count = 0
    def next(self, val: int) -> float:
        self.count += 1
        self.queue.append(val)
        if self.count > self.Wsize:
            out = self.queue.popleft()
        else:
            out = 0
        self.Wsum += + val - out
        return self.Wsum/min(self.Wsize, self.count)
# Your MovingAverage object will be instantiated and called as such:
# obj = MovingAverage(size)
# param_1 = obj.next(val)
'''
############################################################################################################################################################
##############################################################################
##############################################################################
################                   Stack                   ###################
################                   Stack                    ###################
################                   Stack                    ###################
################                   Stack                   ###################
################                   Stack                    ###################
################                   Stack                    ###################
##############################################################################
##############################################################################
##############################################################################
'''
#1209. Remove All Adjacent Duplicates in String II
#s = "deeedbbcccbdaa", k = 3
class Solution:
    def removeDuplicates(self, s: str, k: int) -> str:
        '''  
        #Iterate through the string:
           # if current char is not the same as one before: push 1 to stack
              else we count the repeat, 
           #If the count on the top of the stack equals k, 
           #erase last k characters by poping from the stack.
        '''
        stack = [] # (char, #)
        for i in range(len(s)):
            if not stack or s[i] != stack[-1][0]:
                stack.append((s[i], 1))
            else:
                count = stack.pop()[1] + 1
                if (count % k) != 0:
                    stack.append((s[i], count))
        string = [i[0]*i[1] for i in stack]
        return ''.join(string)





