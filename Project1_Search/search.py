"""
search.py
cs 480 assignment 1: vacuum robot planner

this program implements two search algorithms (uniform-cost search and depth-first search)
to solve planning problems in vacuum world, a grid-based environment where a robot must
clean all dirty cells.

usage:
    python3 search.py [algorithm] [world-file]
    
    [algorithm] should be either 'uniform-cost' or 'depth-first'
    [world-file] should be the path to a .txt file with the grid world
"""
import sys
import heapq
from collections import deque

class VacuumWorld:
    def __init__(self, file_path):
        """load the world from a file, handling various encodings."""
        self.grid = []
        self.robot_start = None
        self.dirty_cells = []
        
        # open file in binary mode to handle potential encoding issues
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # try different decodings
        for encoding in ['utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'latin-1']:
            try:
                text = content.decode(encoding)
                lines = text.splitlines()
                
                # check if we have at least two lines with numbers
                if len(lines) >= 2 and self._has_digit(lines[0]) and self._has_digit(lines[1]):
                    # success! extract dimensions
                    self.cols = self._extract_number(lines[0])
                    self.rows = self._extract_number(lines[1])
                    
                    # extract grid
                    grid_start = 2  # grid starts at line 3 (index 2)
                    for i in range(grid_start, min(grid_start + self.rows, len(lines))):
                        row = self._clean_grid_line(lines[i])
                        if len(row) < self.cols:
                            row = row + '_' * (self.cols - len(row))
                        elif len(row) > self.cols:
                            row = row[:self.cols]
                        self.grid.append(row)
                    
                    # fill in any missing rows
                    while len(self.grid) < self.rows:
                        self.grid.append('_' * self.cols)
                    
                    # find robot and dirty cells
                    self._find_robot_and_dirty_cells()
                    
                    # verify we found the robot
                    if self.robot_start is None:
                        print("warning: no robot start position (@) found in the grid")
                        # just use the first valid cell as robot start for testing
                        for r in range(self.rows):
                            for c in range(self.cols):
                                if not self.is_blocked(r, c):
                                    self.robot_start = (r, c)
                                    print(f"using ({r}, {c}) as robot start position")
                                    break
                            if self.robot_start:
                                break
                    
                    # success!
                    print(f"successfully loaded world with encoding {encoding}")
                    print(f"dimensions: {self.cols}x{self.rows}")
                    print(f"robot start: {self.robot_start}")
                    print(f"dirty cells: {self.dirty_cells}")
                    print("grid:")
                    for row in self.grid:
                        print(row)
                    return
            except Exception as e:
                print(f"failed with encoding {encoding}: {e}")
                continue
        
        # if we get here, we failed to parse the file
        raise ValueError("failed to load world with any encoding")
    
    def _has_digit(self, s):
        """check if a string contains at least one digit."""
        return any(c.isdigit() for c in s)
    
    def _extract_number(self, s):
        """extract the first number from a string."""
        digits = ''
        for c in s:
            if c.isdigit():
                digits += c
        return int(digits) if digits else 0
    
    def _clean_grid_line(self, line):
        """clean a grid line, keeping only valid characters."""
        valid_chars = '_#*@'
        return ''.join(c for c in line if c in valid_chars)
    
    def _find_robot_and_dirty_cells(self):
        """find the robot start position and dirty cells in the grid."""
        for r in range(len(self.grid)):
            for c in range(len(self.grid[r])):
                cell = self.grid[r][c]
                if cell == '@':
                    self.robot_start = (r, c)
                elif cell == '*':
                    self.dirty_cells.append((r, c))
    
    def is_blocked(self, r, c):
        """check if a cell is blocked."""
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
            return True
        if r >= len(self.grid) or c >= len(self.grid[r]):
            return True
        return self.grid[r][c] == '#'
    
    def get_valid_actions(self, r, c):
        """get valid actions (N, S, E, W) from current position."""
        actions = []
        # north
        if not self.is_blocked(r-1, c):
            actions.append(('N', (r-1, c)))
        # south
        if not self.is_blocked(r+1, c):
            actions.append(('S', (r+1, c)))
        # east
        if not self.is_blocked(r, c+1):
            actions.append(('E', (r, c+1)))
        # west
        if not self.is_blocked(r, c-1):
            actions.append(('W', (r, c-1)))
        return actions

def uniform_cost_search(world):
    """implement uniform-cost search to find optimal plan."""
    # a state consists of (robot_r, robot_c, tuple_of_remaining_dirty_cells)
    start_state = (world.robot_start[0], world.robot_start[1], tuple(world.dirty_cells))
    
    # priority queue for ucs
    frontier = [(0, start_state, [])]  # (priority, state, actions_so_far)
    explored = set()
    nodes_generated = 1  # count the start node
    nodes_expanded = 0
    
    while frontier:
        cost, state, actions = heapq.heappop(frontier)
        robot_r, robot_c, dirty_cells = state
        
        # goal test: no dirty cells left
        if not dirty_cells:
            return actions, nodes_generated, nodes_expanded
        
        # skip if this state has been explored
        if state in explored:
            continue
        
        # mark as explored
        explored.add(state)
        nodes_expanded += 1
        
        # check if current position is dirty and can be vacuumed
        current_pos = (robot_r, robot_c)
        if current_pos in dirty_cells:
            # vacuum action
            new_dirty_cells = tuple(cell for cell in dirty_cells if cell != current_pos)
            new_state = (robot_r, robot_c, new_dirty_cells)
            new_actions = actions + ['V']
            heapq.heappush(frontier, (cost + 1, new_state, new_actions))
            nodes_generated += 1
        
        # movement actions
        for action, (new_r, new_c) in world.get_valid_actions(robot_r, robot_c):
            new_state = (new_r, new_c, dirty_cells)
            if new_state not in explored:
                new_actions = actions + [action]
                heapq.heappush(frontier, (cost + 1, new_state, new_actions))
                nodes_generated += 1
    
    # no solution found
    return None, nodes_generated, nodes_expanded

def depth_first_search(world):
    """implement depth-first search to find a plan (not necessarily optimal)."""
    # a state consists of (robot_r, robot_c, tuple_of_remaining_dirty_cells)
    start_state = (world.robot_start[0], world.robot_start[1], tuple(world.dirty_cells))
    
    # stack for dfs
    frontier = [(start_state, [])]  # (state, actions_so_far)
    explored = set()
    nodes_generated = 1  # count the start node
    nodes_expanded = 0
    
    while frontier:
        state, actions = frontier.pop()
        robot_r, robot_c, dirty_cells = state
        
        # goal test: no dirty cells left
        if not dirty_cells:
            return actions, nodes_generated, nodes_expanded
        
        # skip if this state has been explored
        if state in explored:
            continue
        
        # mark as explored
        explored.add(state)
        nodes_expanded += 1
        
        # we'll collect all possible next states
        next_states = []
        
        # check if current position is dirty and can be vacuumed
        current_pos = (robot_r, robot_c)
        if current_pos in dirty_cells:
            # vacuum action
            new_dirty_cells = tuple(cell for cell in dirty_cells if cell != current_pos)
            new_state = (robot_r, robot_c, new_dirty_cells)
            next_states.append((new_state, actions + ['V']))
        
        # movement actions
        for action, (new_r, new_c) in world.get_valid_actions(robot_r, robot_c):
            new_state = (new_r, new_c, dirty_cells)
            if new_state not in explored:
                next_states.append((new_state, actions + [action]))
        
        # add states to frontier in reverse order (so we explore depth-first)
        for new_state, new_actions in reversed(next_states):
            frontier.append((new_state, new_actions))
            nodes_generated += 1
    
    # no solution found
    return None, nodes_generated, nodes_expanded

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 search.py [algorithm] [world-file]")
        sys.exit(1)
    
    algorithm = sys.argv[1]
    world_file = sys.argv[2]
    
    try:
        # load the world
        world = VacuumWorld(world_file)
        
        # check if we have any dirty cells
        if not world.dirty_cells:
            print("Warning: No dirty cells (*) found in the grid. Solution will be empty.")
        
        # run the appropriate search algorithm
        if algorithm == 'uniform-cost':
            plan, nodes_generated, nodes_expanded = uniform_cost_search(world)
        elif algorithm == 'depth-first':
            plan, nodes_generated, nodes_expanded = depth_first_search(world)
        else:
            print("Error: Algorithm must be either 'uniform-cost' or 'depth-first'")
            sys.exit(1)
        
        # print the result
        if plan:
            for action in plan:
                print(action)
            print(f"{nodes_generated} nodes generated")
            print(f"{nodes_expanded} nodes expanded")
        else:
            print("No solution found")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()