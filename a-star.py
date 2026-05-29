import heapq

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    
    heap = []
    heapq.heappush(heap, (0, start))
    
    came_from = {}
    g = {start: 0}
    
    while heap:
        _, current = heapq.heappop(heap)
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = current[0] + dx, current[1] + dy
            
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0:
                neighbor = (nx, ny)
                new_g = g[current] + 1
                
                if neighbor not in g or new_g < g[neighbor]:
                    g[neighbor] = new_g
                    f = new_g + heuristic(neighbor, goal)
                    heapq.heappush(heap, (f, neighbor))
                    came_from[neighbor] = current
    
    return None


# Example
grid = [
    [0,1,0,0,0,0],
    [0,0,0,1,1,0],
    [0,0,0,1,0,0],
    [0,1,1,0,0,0],
    [0,1,0,0,0,0],
    [0,0,1,0,0,0]
]

start = (0,0)
goal = (5,5)

print("Path:", a_star(grid, start, goal))