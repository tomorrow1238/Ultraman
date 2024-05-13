import argparse
import os
import pymeshlab
import trimesh
import numpy as np

def decimation_mesh(input_path, output_path,target_perc=0.1):
    mesh = trimesh.load_mesh(input_path, force='mesh')
    xyz = mesh.vertices
    faces= mesh.faces
    rgb = mesh.visual.vertex_colors.astype(float)/255.
    ms = pymeshlab.Mesh(vertex_matrix = xyz, face_matrix = faces, v_color_matrix=rgb)
    mss = pymeshlab.MeshSet()
    mss.add_mesh(ms)
    mss.meshing_decimation_quadric_edge_collapse(targetperc=target_perc)
    m  = mss.mesh(0)
    v  = m.vertex_matrix()
    vn = m.vertex_normal_matrix()
    vc = m.vertex_color_matrix()
    f  = m.face_matrix()        
    x_min, y_min, z_min = np.argmin(v, axis=0)
    x_max, y_max, z_max = np.argmax(v, axis=0)
    if sum([vn[x_min][0]>0, vn[y_min][1]>0, vn[z_min][2]>0, vn[x_max][0]<0, vn[y_max][1]<0, vn[z_max][2]<0]) > 3:
        vn *= -1.
    ex_mesh = trimesh.Trimesh(vertices=v, faces=f, vertex_normals=vn, vertex_colors=vc)
    save_path = os.path.join(output_path,'decimation_mesh.obj')
    ex_mesh.export(save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    decimation_mesh(args.input_path, args.output_path)