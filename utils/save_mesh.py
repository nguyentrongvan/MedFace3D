def save_ply(points, file_path):
    header = """ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
end_header
""".format(len(points))

    with open(file_path, 'w') as f:
        f.write(header)
        for point in points:
            x, y, z = point
            f.write("{} {} {}\n".format(x, y, z))


def save_ply_mesh(vertices, faces, file_path):
    header = """ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
element face {}
property list uchar int vertex_indices
end_header
""".format(len(vertices), len(faces))

    with open(file_path, 'w') as f:
        f.write(header)

        for vertex in vertices:
            x, y, z = vertex
            f.write("{} {} {}\n".format(x, y, z))

        for face in faces:
            f.write("3 {} {} {}\n".format(face[0], face[1], face[2]))