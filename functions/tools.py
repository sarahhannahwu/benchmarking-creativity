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
        # Convert the list of bits into a binary string
        binary_str = ''.join([str(bit) for bit in row])

        # Convert the binary string into an integer
        number = int(binary_str, 2)

        codes.append(str(number))

    return ' '.join(codes)