import trimesh

import torch

import sys
sys.path.append(".")

import xatlas
from PIL import Image
from torchvision import transforms
from lib.uv_helper import Pytorch3dRasterizer
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import rasterize_meshes

import os
import argparse
import os.path as osp
import numpy as np
import pdb

# get cuda
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    torch.cuda.set_device(DEVICE)
else:
    print("no gpu avaiable")
    exit()

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--obj_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    return args


def export_obj(v_np, f_np, vt, ft, path):

    # write mtl info into obj
    new_line = f"mtllib mesh.mtl \n"
    vt_lines = "\nusemtl mesh \n"
    v_lines = ""
    f_lines = ""

    for _v in v_np:
        v_lines += f"v {_v[0]} {_v[1]} {_v[2]}\n"
    for fid, _f in enumerate(f_np):
        f_lines += f"f {_f[0]+1}/{ft[fid][0]+1} {_f[1]+1}/{ft[fid][1]+1} {_f[2]+1}/{ft[fid][2]+1}\n"
    for _vt in vt:
        vt_lines += f"vt {_vt[0]} {_vt[1]}\n"
    new_file_data = new_line + v_lines + vt_lines + f_lines
    with open(path, 'w') as file:
        file.write(new_file_data)


if __name__ == "__main__":
    args = init_args()
    
    output_dir = osp.join(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Load the mesh")
    model_path = osp.join(args.input_dir, "{}.obj".format(args.obj_name))
    mesh = trimesh.load(model_path)
    vertices, faces = mesh.vertices, mesh.faces

    print("Get the color")
    color = mesh.visual.vertex_colors[:, :3]

    ply_path = osp.join(args.output_dir, 'uv.ply')
    mesh.export(ply_path)
    # pdb.set_trace()

    print("UV texture rendering")

    v_np = vertices*100
    f_np = faces

    vt_cache = osp.join(args.output_dir, 'vt.pt')
    ft_cache = osp.join(args.output_dir, 'ft.pt')

    atlas = xatlas.Atlas()
    atlas.add_mesh(v_np, f_np)
    print("1")
    chart_options = xatlas.ChartOptions()
    pack_options = xatlas.PackOptions()
    chart_options.max_iterations = 4
    chart_options.texture_seam_weight = 2.0
    pack_options.resolution = 8192
    pack_options.bruteForce = True
    atlas.generate(chart_options=chart_options)
    print("2")
    vmapping, ft_np, vt_np = atlas[0]

    vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(DEVICE)
    ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(DEVICE)
    torch.save(vt.cpu(), vt_cache)
    torch.save(ft.cpu(), ft_cache)

    # vt = torch.load(vt_cache).to(DEVICE)
    # ft = torch.load(ft_cache).to(DEVICE)

    print("ready for UV rasterizer")
    uv_rasterizer = Pytorch3dRasterizer(image_size=8192, device=DEVICE)
    print("3")
    texture_npy = uv_rasterizer.get_texture(
        torch.cat([(vt - 0.5) * 2.0, torch.ones_like(vt[:, :1])], dim=1),
        ft,
        torch.tensor(v_np).unsqueeze(0).float(),
        torch.tensor(f_np).unsqueeze(0).long(),
        torch.tensor(color).unsqueeze(0).float() / 255.0,
    )
    print("4")

    gray_texture = texture_npy.copy()
    mask = gray_texture.sum(axis=2) == 0.0
    gray_texture[mask] = 0.5

    print("export the texture file")
    texture_path = osp.join(args.output_dir, 'mesh.png')
    Image.fromarray((gray_texture * 255.0).astype(np.uint8)).save(texture_path)

    print("export the mtl file")
    with open(f"{output_dir}/mesh.mtl", "w") as fp:
        fp.write(f"newmtl mesh \n")
        fp.write(f"Ka 1.000000 1.000000 1.000000 \n")
        fp.write(f"Kd 1.000000 1.000000 1.000000 \n")
        fp.write(f"Ks 0.000000 0.000000 0.000000 \n")
        fp.write(f"Tr 1.000000 \n")
        fp.write(f"illum 1 \n")
        fp.write(f"Ns 0.000000 \n")
        fp.write(f"map_Kd mesh.png \n")

    print("export the mesh file")
    export_obj(v_np, f_np, vt, ft, f"{output_dir}/mesh.obj")