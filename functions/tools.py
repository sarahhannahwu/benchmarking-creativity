import pandas as pd

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