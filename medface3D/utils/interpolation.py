import numpy as np

def bilinear_interpolation_triangle(A, B, C, u = 1/3, v = 1/3):
    # Calculate the weights for each vertex of the triangle
    w_A = u
    w_B = v
    w_C = 1 - u - v

    # Calculate the interpolated point using the barycentric coordinates
    x_interpolated = w_A * A[0] + w_B * B[0] + w_C * C[0]
    y_interpolated = w_A * A[1] + w_B * B[1] + w_C * C[1]

    return (int(x_interpolated), int(y_interpolated))


def bilinear_interpolation(list_values, u = 1/3, v = 1/3):
    if len(list_values) != 3:
        raise ValueError("Input list_values must contain exactly 3 values.")

    w_A = u
    w_B = v
    w_C = 1 - u - v

    x1, x2, x3 = list_values

    pt_interpolated = w_A * x1 + w_B * x2 + w_C * x3
    return pt_interpolated