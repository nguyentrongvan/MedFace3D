def bilinear_interpolation_triangle(A, B, C, u=0.3, v=0.4):
    # Calculate the weights for each vertex of the triangle
    w_A = u
    w_B = v
    w_C = 1 - u - v

    # Calculate the interpolated point using the barycentric coordinates
    x_interpolated = w_A * A[0] + w_B * B[0] + w_C * C[0]
    y_interpolated = w_A * A[1] + w_B * B[1] + w_C * C[1]

    return (int(x_interpolated), int(y_interpolated))


def bilinear_interpolation(list_value, w = 0.4, h = 0.6):
    x1, x2, x3 = list_value
    # Calculate the top interpolation (linear interpolation along the width)
    top_interp = (1 - w) * x1 + w * x2
    # Calculate the bottom interpolation (linear interpolation along the width)
    bottom_interp = (1 - w) * x3 + w * x1
    # Calculate the final result (linear interpolation along the height)
    result = (1 - h) * top_interp + h * bottom_interp
    return result
