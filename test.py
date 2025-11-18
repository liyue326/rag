# 贪婪算法 分饼干
def findContentChildren(g: list, s: list) -> int:  
    g.sort()  
    s.sort()  
    child = cookie = 0  
    while child < len(g) and cookie < len(s):  
        if s[cookie] >= g[child]:  
            child += 1  # 孩子被满足  
        cookie += 1     # 无论是否满足，饼干被消耗  
    return child      

print(findMaxChild([1,3,2,5,6],[1,2,3,5]))

#双指针 两数和
def findTwo(arr: list[int],target:int) -> list[int]:
  i,j=0,len(arr)-1
  while i<j:
    if(arr[i] + arr[j]==target):
      break;
    if(arr[i]+arr[j]<target): 
      i+=1
    else: j-=1
  return [i,j]

findTwo([2,7,11,15],9) 


#二分查找 第一次出现的位置和最后一次出现的位置
def findFirstLastPos(arr:list[int], target: int) -> list[int]:
   left,right = 0, len(arr)-1
   first= -1
   while left<=right:
      mid=(left+right)//2
      if(arr[mid] == target):
         first = mid 
      if(arr[mid] < target):     # 查找左边界时收缩右边界
         left=mid+1
      else: 
         right=mid-1   
      
   
   left,right = 0, len(arr)-1
   last= -1
   while left<=right:
      mid=(left+right)//2
      if(arr[mid] == target):
         last =  mid 
      if(arr[mid] > target):  # 查找右边界时收缩左边界
         right=mid-1   
      else: 
         left=mid+1
      
   return [first, last]

findFirstLastPos([1,3,5,8,8,9,10],8)   


# 动态规划DP：爬楼梯
def climbStairs(n):
    if n == 0:
        return 1
    elif n == 1:
        return 1
    dp = [0] * (n + 1)
    dp[0] = 1  # 初始状态
    dp[1] = 1  # 到达第1阶的方法数
    dp[2] = 2  # 到达第2阶的方法数
    for i in range(3, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

#假设你正在一个 m x n 的网格中，从左上角 (0, 0) 开始，每次只能向右或向下移动一步，问有多少种不同的路径可以到达右下角 (m-1, n-1)。
def unique_paths(m, n):
    # 创建一个 m x n 的二维数组，初始化为1
    dp = [[1] * n for _ in range(m)]
    # 填充 dp 数组
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    # 返回右下角的值
    return dp[m-1][n-1]

#深度优先（DFS）最大岛屿面积
def maxAreaOfIsland(grid):
    if not grid:
        return 0
    max_area = 0
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    def dfs(i, j):
        if 0 <= i < rows and 0 <= j < cols and grid[i][j] == 1:
            grid[i][j] = 0  # Mark as visited
            return 1 + dfs(i-1, j) + dfs(i+1, j) + dfs(i, j-1) + dfs(i, j+1)
        return 0
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 1:
                max_area = max(max_area, dfs(i, j))
    return max_area
grid = [
    [1, 0, 1, 1, 0, 1, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 1]
]
maxAreaOfIsland(grid)


# 翻转字符串
def reverse(str:list):
  left,right = 0,len(str)-1
  while left<right:
   str[left],str[right] = str[right],str[left]
   left+=1
   right-=1


# 最长字符串 滑动窗口
def maxSize(str):
   maxLength =0
   set = set()
   left = 0
   for right in range(len(str)):
      while  str[right] in set:
         set.remove(str[left])
         left+=1
      set.add(str[right])
   return max(maxLength,right-left+1)      