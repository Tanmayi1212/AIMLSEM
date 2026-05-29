import heapq
def heuristic(state,goal):
    dist=0
    for i in range(9):
        if state[i]!=0:
            gi=goal.index(state[i])
            dist+=abs(i//3-gi//3)+abs(i%3-gi%3)
    return dist
    
def get_nei(state):
    res=[]
    i=state.index(0)
    x,y=i//3,i%3
    
    for dx,dy in [(0,1),(0,-1),(1,0),(-1,0)]:
        nx,ny=x+dx,y+dy
        if 0<=nx<3 and 0<=ny<3:
            new=list(state)
            ni=nx*3+ny
            new[i],new[ni]=new[ni],new[i]
            res.append(tuple(new))
    return res
    
def a_star(start,goal):
    heap=[]
    heapq.heappush(heap,(heuristic(start,goal),0,start,[]))
    visited=set()
    
    while heap:
        _,g,state,path=heapq.heappop(heap)
        
        if state==goal:
            return path
        visited.add(state)
            
        for nei in get_nei(list(state)):
            if nei not in visited:
                heapq.heappush(heap,(g+1+heuristic(nei,goal),g+1,nei,path+[nei]))
    return None
    
def print_state(state):
    for i in range(0,9,3):
        print(state[i:i+3])
    print()
    
start=(1,2,3,4,5,6,7,0,8)
goal=(1,2,3,4,5,6,0,7,8)

ans=a_star(start,goal)

for step,state in enumerate(ans):
    print(f"Step {step}")
    print_state(state)