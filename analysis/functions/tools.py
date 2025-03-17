import pandas as pd
from pydantic import BaseModel
import copy

class GameMove(BaseModel):
    shape: str
    next_shape: str
    response: str 

def decode_shape_binaries_str(encoded_str, bits=10):
    """
    Decodes a single string of space-separated decimal codes into
    a 2D list (shape) of 0/1 bits. Each code becomes one row in the shape.

    :param encoded_str: A single string with space-separated decimal values
                        (e.g., "1016 64 64 64").
    :param bits: The fixed width of the binary representation (default=10).
    :return: A list of lists, where each sub-list is a row of bits (0's and 1's).
    """
    # Split the string by spaces to get each code as a separate token
    codes = encoded_str.split()

    shape = []
    for code in codes:
        
        # Convert the code (string) to an integer
        number = int(code)

        # Convert to binary, left-padded with zeros to the desired bit length
        binary_str = format(number, 'b').rjust(bits, '0')

        # Convert the binary string into a list of integer bits (0 or 1)
        row_of_bits = [int(bit) for bit in binary_str]
        shape.append(row_of_bits)

    return shape

def encode_shape_binaries(shape, bits=10):
    """
    Encodes a 2D list (shape) of 0/1 bits into a single string of
    space-separated decimal codes. Each row in the shape becomes one code.

    :param shape: A list of lists, where each sub-list is a row of bits (0's and 1's).
    :param bits: The fixed width of the binary representation (default=10).
    :return: A single string with space-separated decimal values
             (e.g., "1016 64 64 64").
    """
    codes = []
    for row in shape:
        if 1 not in row:
            continue
        # Convert the list of bits into a binary string
        binary_str = ''.join([str(bit) for bit in row])

        # Convert the binary string into an integer
        number = int(binary_str, 2)

        codes.append(str(number))

    return ' '.join(codes)

def matrix_to_coordinates(matrix):
    """
    Convert a 2D array of bits to a list of (x,y) coordinate tuples on a 10x10 grid.
    
    Args:
        matrix: A 2D array/list where 1 represents a filled square and 0 represents an empty square
        
    Returns:
        A list of (x,y) tuples representing the coordinates of filled squares
        where (0,0) is the top-left corner
    """
    coordinates = []
    
    # Check if matrix is provided as a string representation
    if isinstance(matrix, str):
        import ast
        try:
            # Try to parse as a Python literal (list of lists)
            matrix = ast.literal_eval(matrix)
        except:
            # If that fails, try processing as a string representation
            matrix = matrix.replace('[', '').replace(']', '')
            rows = matrix.split('],[')
            parsed_matrix = []
            for row in rows:
                parsed_matrix.append([int(cell) for cell in row.split(',')])
            matrix = parsed_matrix
    
    # Iterate through the matrix and collect coordinates of filled cells
    for y, row in enumerate(matrix):
        for x, cell in enumerate(row):
            if cell == 1:
                coordinates.append((x, y))
    
    return coordinates

def is_contiguous(shape):
    rows = len(shape)
    cols = len(shape[0])
    visited = set()

    if sum(sum(row) for row in shape) == 10:  # Check if the total number of 1s is 10
        # Determine the starting point, which is the first 1 in the shape
        for r in range(rows):
            for c in range(cols):
                if shape[r][c] == 1:
                    start = (r, c)
                    break
        def dfs(r, c):
            if (r, c) in visited or not (0 <= r < rows and 0 <= c < cols) or shape[r][c] == 0:
                return 0
            visited.add((r, c))
            count = 1 
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]: # Check all 4 directions
                count += dfs(r + dr, c + dc)
            return count
        
        return dfs(start[0], start[1]) == 10
    return False

def is_one_change(shape, next_shape):
    rows_shape = len(shape)
    rows_next_shape = len(next_shape)
    cols = len(shape[0])
    
    # Find the row with the greatest overlap in the number of 1s
    max_overlap = 0
    best_offset = 0
    for offset in range(-rows_shape + 1, rows_next_shape):
        overlap = 0
        for r in range(rows_shape):
            if 0 <= r + offset < rows_next_shape:
                overlap += sum(1 for c in range(cols) if shape[r][c] == 1 and next_shape[r + offset][c] == 1)
        if overlap > max_overlap:
            max_overlap = overlap
            best_offset = offset
    
    # Add a row of 0s to the shape with fewer rows
    if rows_shape < rows_next_shape:
        if best_offset > 0:
            shape = [[0] * cols] * best_offset + shape
        else:
            shape = shape + [[0] * cols] * (-best_offset)
    elif rows_shape > rows_next_shape:
        if best_offset > 0:
            next_shape = [[0] * cols] * best_offset + next_shape
        else:
            next_shape = next_shape + [[0] * cols] * (-best_offset)

    change_count = 0
    for r in range(len(shape)):
        for c in range(len(shape[0])):
            if shape[r][c] != next_shape[r][c]:
                change_count += 1
    return change_count == 2  # Exactly one change (one removal and one addition)

def is_valid_move(shape, next_shape): # Take in the shape and next_shape as 2D lists
    if is_contiguous(next_shape) and is_one_change(shape, next_shape):
        return True
    return False


def neighbors(polyomino):
    """
    ARGS: polyomino: a list of tuples representing the squares of a polyomino
    RETURN: a list of tuples representing the squares that are valid neighbors of the polyomino
    """
    found = []
    for (x,y) in polyomino:
        for delta_x, delta_y in [(1,0), (-1,0), (0,1), (0,-1)]:
            if y + delta_y < 0 or (y + delta_y == 0 and x + delta_x < 0):
                continue
            new_square = (x + delta_x, y + delta_y)
            if new_square not in found:
                found.append(new_square)
    return found

def redelmeier(n):
    """
    ARGS: n: the size of the polyominoes to count
    Caller function to redelmeier_recursion. Initializes the counts list and calls the recursive function.
    """
    polyomino = []
    n_ominoes = []
    untried_set = [(0,0)]
    redelmeier_recursion(n, n_ominoes, polyomino, untried_set)
    return n_ominoes

def redelmeier_recursion(n, n_ominoes, polyomino, untried_set):
    while len(untried_set) > 0:
        new_square = untried_set.pop()
        new_untried_set = copy.copy(untried_set)
        new_square_neighbors = neighbors([new_square])
        polyomino_neighbors = neighbors(polyomino)
        for s in new_square_neighbors:
            if s not in polyomino_neighbors and s not in polyomino:
                new_untried_set.append(s)
        new_polyomino = copy.copy(polyomino)
        new_polyomino.append(new_square)
        if len(new_polyomino) < n:
            redelmeier_recursion(n, n_ominoes, new_polyomino, new_untried_set)
        else:
            n_ominoes.append(new_polyomino)
            


# Converts tuple representation of polyomino to grid representation
def tuple_polyomino_to_grid(tuple_polyomino):
    min_x = min([x for (x,y) in tuple_polyomino])
    min_y = min([y for (x,y) in tuple_polyomino])
    tuple_polyomino = [(x - min_x, y - min_y) for (x,y) in tuple_polyomino]
    grid = [[0 for i in range(10)] for j in range(10)]
    for (x,y) in tuple_polyomino:
        grid[x][y] = 1 # change to [y][x] if you want the shape top-justified
    return grid

def get_rotations(decomino):
    rotate = lambda decomino: [(y,-x) for (x,y) in decomino]
    rotations = []
    for _ in range(4):      
        decomino = rotate(decomino)
        rotations.append(decomino)
    return rotations


def get_reflections(decomino):
    return [[(-x,y) for (x,y) in decomino], [(x,-y) for (x,y) in decomino]]

def get_all_transformations(decomino):
    """Get all possible transformations (rotations and reflections) of a decomino"""
    transformations = []
    for rotation in get_rotations(decomino):
        transformations.append(rotation)
        for reflection in get_reflections(rotation):
            transformations.append(reflection)
    return transformations

# Takes in the fixed decominoes and returns the smaller set of free decominoes
def get_unique_decominoes(decominoes):
    unique_decominoes = set()
    for decomino in decominoes:
        transformations = [] # A list of tuple representations of the decomino's rotations and reflections
        for rotation in get_rotations(decomino):
            transformations.append(rotation)
            for reflection in get_reflections(rotation):
                transformations.append(reflection)
        transformations_grids = map(tuple_polyomino_to_grid, transformations)
        transformation_encodings_set = set(map(encode_shape_binaries, transformations_grids))
        if not transformation_encodings_set.intersection(unique_decominoes): # Transform every possible way and check if any of them are already in the set
            unique_decominoes.add(encode_shape_binaries(tuple_polyomino_to_grid(decomino)))
    return unique_decominoes

# Takes in the fixed decominoes and returns the smaller set of free decominoes
def print_unique_decominoes(decominoes):
    unique_decominoes = set()
    for decomino in decominoes:
        transformations = [] # A list of tuple representations of the decomino's rotations and reflections
        for rotation in get_rotations(decomino):
            transformations.append(rotation)
            for reflection in get_reflections(rotation):
                transformations.append(reflection)
        transformations_grids = map(tuple_polyomino_to_grid, transformations)
        transformation_encodings_set = set(map(encode_shape_binaries, transformations_grids)) 
        print(f"Decomino: {decomino}")
        print(f"Transformation Encodings: {transformation_encodings_set}")
        if not transformation_encodings_set.intersection(unique_decominoes): # Transform every possible way and check if any of them are already in the set
            unique_decominoes.add(encode_shape_binaries(tuple_polyomino_to_grid(decomino)))
    return unique_decominoes

def grid_to_string(grid):
    string = ""
    for row in grid:
        # Convert each number to string before joining
        string += " ".join(str(cell) for cell in row).replace('1', '■').replace('0', '_') + "\n"
    return string