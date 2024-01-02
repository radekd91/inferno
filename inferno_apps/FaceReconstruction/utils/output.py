import inferno 
from pathlib import Path
import numpy as np 
import inferno.utils.DecaUtils as util
from skimage.io import imsave
from inferno.utils.lightning_logging import _fix_image


def torch_img_to_np(img):
    if isinstance(img, np.ndarray): 
        return img
    return img.detach().cpu().numpy().transpose(1, 2, 0)


def save_obj(face_rec_model, filename, opdict, i=0):
    # dense_template_path = Path(inferno.__file__).parents[1] / 'assets' / "DECA" / "data" / 'texture_data_256.npy'
    vertices = opdict['verts'][i].detach().cpu().numpy()
    faces = face_rec_model.renderer.render.faces[0].detach().cpu().numpy()
    uvcoords = face_rec_model.renderer.render.raw_uvcoords[0].detach().cpu().numpy()
    uvfaces = face_rec_model.renderer.render.uvfaces[0].detach().cpu().numpy()
    # save coarse mesh, with texture and normal map
    util.write_obj(filename, vertices, faces,
                #    texture=texture,
                   uvcoords=uvcoords,
                   uvfaces=uvfaces,
                #    normal_map=normal_map
                   )


def save_images(outfolder, name, vis_dict, i = 0, with_detection=False):
    prefix = None
    final_out_folder = Path(outfolder) / name
    final_out_folder.mkdir(parents=True, exist_ok=True)

    if with_detection:
        imsave(final_out_folder / f"inputs.png", vis_dict['image'][i])
    imsave(final_out_folder / f"geometry.png",  vis_dict['shape_image'][i])
    # imsave(final_out_folder / f"out_im.png", vis_dict['predicted_image'][i])
    


def save_codes(output_folder, name, vals, i = None):
    if i is None:
        np.save(output_folder / name / f"shape.npy", vals["shapecode"].detach().cpu().numpy())
        np.save(output_folder / name / f"exp.npy", vals["expcode"].detach().cpu().numpy())
        np.save(output_folder / name / f"tex.npy", vals["texcode"].detach().cpu().numpy())
        np.save(output_folder / name / f"jawpose.npy", vals["jawpose"].detach().cpu().numpy())
        np.save(output_folder / name / f"globalpose.npy", vals["globalpose"].detach().cpu().numpy())
        np.save(output_folder / name / f"cam.npy", vals["cam"].detach().cpu().numpy())
        np.save(output_folder / name / f"lightcode.npy", vals["lightcode"].detach().cpu().numpy())
    else: 
        np.save(output_folder / name / f"shape.npy", vals["shapecode"][i].detach().cpu().numpy())
        np.save(output_folder / name / f"exp.npy", vals["expcode"][i].detach().cpu().numpy())
        np.save(output_folder / name / f"tex.npy", vals["texcode"][i].detach().cpu().numpy())
        np.save(output_folder / name / f"jawpose.npy", vals["jawpose"][i].detach().cpu().numpy())
        np.save(output_folder / name / f"globalpose.npy", vals["globalpose"][i].detach().cpu().numpy())
        np.save(output_folder / name / f"cam.npy", vals["cam"][i].detach().cpu().numpy())
        np.save(output_folder / name / f"lightcode.npy", vals["lightcode"][i].detach().cpu().numpy())

