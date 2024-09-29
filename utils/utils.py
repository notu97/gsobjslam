import yaml
import numpy as np
import torch
import open3d as o3d
from gaussian_rasterizer import GaussianRasterizationSettings, GaussianRasterizer
from gaussian_model import GaussianModel
import pickle
from scipy.spatial import ConvexHull



def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def np2torch(array: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """Converts a NumPy ndarray to a PyTorch tensor.
    Args:
        array: The NumPy ndarray to convert.
        device: The device to which the tensor is sent. Defaults to 'cpu'.

    Returns:
        A PyTorch tensor with the same data as the input array.
    """
    return torch.from_numpy(array).float().to(device)


def torch2np(tensor: torch.Tensor) -> np.ndarray:
    """ Converts a PyTorch tensor to a NumPy ndarray.
    Args:
        tensor: The PyTorch tensor to convert.
    Returns:
        A NumPy ndarray with the same data and dtype as the input tensor.
    """
    return tensor.detach().cpu().numpy()


def get_render_settings(w, h, intrinsics, w2c: np.ndarray, near=0.01, far=100, sh_degree=0):
    """
    Constructs and returns a GaussianRasterizationSettings object for rendering,
    configured with given camera parameters.

    Args:
        width (int): The width of the image.
        height (int): The height of the image.
        intrinsic (array): 3*3, Intrinsic camera matrix.
        w2c (array): World to camera transformation matrix.
        near (float, optional): The near plane for the camera. Defaults to 0.01.
        far (float, optional): The far plane for the camera. Defaults to 100.

    Returns:
        GaussianRasterizationSettings: Configured settings for Gaussian rasterization.
    """
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1,
                                                  1], intrinsics[0, 2], intrinsics[1, 2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    viewmatrix = w2c.transpose(0, 1)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far /
                                    (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]], device='cuda').float().transpose(0, 1)
    full_proj_matrix = viewmatrix.unsqueeze(
        0).bmm(opengl_proj.unsqueeze(0)).squeeze(0)
    return GaussianRasterizationSettings(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], device='cuda').float(),
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=full_proj_matrix,
        sh_degree=sh_degree,
        campos=cam_center,
        prefiltered=False,
        debug=False)


def np2ptcloud(pts: np.ndarray, rgb=None) -> o3d.geometry.PointCloud:
    """converts numpy array to point cloud
    Args:
        pts (ndarray): point cloud
    Returns:
        (PointCloud): resulting point cloud
    """
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    if rgb is not None:
        cloud.colors = o3d.utility.Vector3dVector(rgb)
    return cloud


def render_gs(gaussian_models: list, render_settings: object) -> dict:

    renderer = GaussianRasterizer(raster_settings=render_settings)

    means3D = torch.nn.Parameter(data=torch.vstack([model.get_xyz() for model in gaussian_models]))
    means2D = torch.zeros_like(
        means3D, dtype=means3D.dtype, requires_grad=True, device="cuda")
    means2D.retain_grad()
    opacities = torch.vstack([model.get_opacity() for model in gaussian_models])

    shs, colors_precomp = None, None
    shs = torch.concatenate([model.get_features() for model in gaussian_models], dim=0)
    scales = torch.vstack([model.get_scaling() for model in gaussian_models])
    rotations = torch.vstack([model.get_rotation() for model in gaussian_models])

    render_args = {
        "means3D": means3D,
        "means2D": means2D,
        "opacities": opacities,
        "colors_precomp": colors_precomp,
        "shs": shs,
        "scales": scales,
        "rotations": rotations,
        "cov3D_precomp": None
    }
    color, depth, alpha, radii = renderer(**render_args)

    return {"color": color, "depth": depth, "radii": radii, "means2D": means2D, "alpha": alpha}

def Projection_jacobian(mean_3D):

    J = torch.tensor([[1/mean_3D[2], 0, -mean_3D[0]/mean_3D[2]**2],
                      [0, 1/mean_3D[2], -mean_3D[1]/mean_3D[2]**2],
                      [0 , 0, 0]])
    J = J.to('cuda:0')
    return J

def pi_func(v):
    '''
    projection function
    v is in homogenous coords
    '''
    pi = (torch.matmul(torch.eye(3,4, device = 'cuda:0'), v))/v[2]
    pi = pi.to('cuda:0')
    # print(pi)
    return pi 

def image_projected_submap(gaussian_model: GaussianModel, keyframe: dict) -> dict:


    means3D = gaussian_model.get_xyz()
    covs3D = gaussian_model.get_actual_covariance()

    print("covs3D")
    # print(covs3D)

    with open("covs3D", "wb") as fp:   #Pickling
            pickle.dump(covs3D, fp)

    size = means3D.shape[0]
    print("size: ", size)

    ones = torch.ones((size,1), device='cuda:0')

    homogenous_means3D = torch.hstack((means3D,ones)).detach()
    # print("homogenous_means3D: ",homogenous_means3D)
    input("press a key: 4")

    W2c =  torch.inverse( np2torch(keyframe['c2w'], device='cuda:0')) # World to Camera
    R = W2c[0:3,0:3] # Rotation Matrix
    print('Rotation: ',R)

    means2D = torch.zeros(size, 2, device = 'cuda:0')
    means_cam = torch.matmul(W2c,homogenous_means3D.T)

    # print("Means Cam: ",means_cam)
    input("press a key: 5")


    for i in range(size):
         means2D[i,:] = pi_func(means_cam[:,i])[0:2]
    
    # print("Means2D: ",means2D)
    
    with open("means2D", "wb") as fp:   #Pickling
        pickle.dump(means2D, fp)

    input("press a key: 6")

    covs2D = torch.zeros(3*size, 3, device = 'cuda:0')

    for i in range(size):
        cov3d = covs3D[i].detach()
        mean3d = means3D[i].detach()

        jr = torch.matmul(Projection_jacobian(mean3d),R)

        # print(f'jr: {jr}')
        # print(f'cov3d: {cov3d}')
        covs2D[3*i:(3*i)+3,0:3] = torch.matmul(jr,torch.matmul(cov3d,jr.T))
    
    print("covs2D ")
    # print(covs2D)

    with open("covs2D", "wb") as fp:   #Pickling
            pickle.dump(covs2D, fp)

    input("press a key: 7")

    print(" Size of actual Covaraince from GS Model: ", gaussian_model.get_actual_covariance().shape)
    print("   Size of 3D Mean from GS Model: ", gaussian_model.get_xyz()[0])
    # print("    Size of 2D Mean from GS Model: ", means2D[0])

    means2D_list = []
    for mean in means2D.cpu().numpy():
        means2D_list.append(mean)

    covs_list = []
    for i in range(size):
        cov = covs2D[3*i:(3*i)+3,0:3].cpu().numpy()
        # print(cov[0:2,0:2])
        covs_list.append(cov[0:2,0:2])

    return {"means2D": means2D, "covs2D": covs2D, "means2D_list": means2D_list, "covs_list": covs_list }


# Function to generate points on the boundary of a 2D Gaussian
def gaussian_ellipse(mean, cov, n_points=100, confidence_level=5.991):
    t = np.linspace(0, 2 * np.pi, n_points)
    circle = np.array([np.cos(t), np.sin(t)])  # Unit circle points
    eigvals, eigvecs = np.linalg.eigh(cov)  # Eigen decomposition of the covariance
    axis_lengths = np.sqrt(eigvals * confidence_level)  # Scale according to confidence level
    ellipse = eigvecs @ np.diag(axis_lengths) @ circle  # Rotate and scale the unit circle
    return ellipse.T + mean  # Translate to the mean

def map_coordinates_to_pixels(x, y, width=640, height=480):
    """
    Maps coordinates (x, y) in the range [-0.5, 0.5] to pixel values in a 640x480 image.
    
    Args:
    x (float): x-coordinate in the range [-0.5, 0.5].
    y (float): y-coordinate in the range [-0.5, 0.5].
    width (int): Width of the image (default is 640).
    height (int): Height of the image (default is 480).
    
    Returns:
    (int, int): Corresponding (x_pixel, y_pixel) in the image.
    """
    # Map x from [-0.5, 0.5] to [0, width-1]
    x_pixel = int((x + 0.5) * width)
    
    # Map y from [-0.5, 0.5] to [0, height-1]
    y_pixel = int((y + 0.5) * height)
    
    # Ensure the pixel values are within image bounds
    x_pixel = max(0, min(x_pixel, width - 1))
    y_pixel = max(0, min(y_pixel, height - 1))
    
    return x_pixel, y_pixel