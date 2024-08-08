
import torch
import numpy as np

from torch import nn


from utils.graphics_utils import MeshPointCloud
from utils.sh_utils import RGB2SH
from utils.general_utils import inverse_sigmoid, rot_to_quat_batch
from scene.gaussian_model import GaussianModel


class GaussianMeshModel(GaussianModel):
    def __init__(self, sh_degree: int, use_img_feats: bool = False):
        super().__init__(sh_degree, use_img_feats)
        self.vertices = torch.empty(0)
        self.faces = torch.empty(0)
    
    def create_from_pcd(self, pcd: MeshPointCloud, spatial_lr_scale: float):
        print("Creating GaussianMeshModel from MeshPointCloud")
        self.spatial_lr_scale = spatial_lr_scale


        pcd_alpha_shape = pcd.alpha.shape
        print("Number of faces: ", pcd_alpha_shape[0])
        print("Number of points at initialisation in face: ", pcd_alpha_shape[1])

        alpha_point_cloud = pcd.alpha.float().cuda()
        scale = torch.ones((pcd.points.shape[0], 1)).float().cuda()

        print("Number of points at initialisation : ",
            alpha_point_cloud.shape[0] * alpha_point_cloud.shape[1])

        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        opacities = inverse_sigmoid(0.1 * torch.ones((pcd.points.shape[0], 1), dtype=torch.float, device="cuda"))

        ## use alpha, vertex, face to init self._xyz
        alpha = self.update_alpha(pcd) ## _alpha means a1,a2,a3 in paper, controlling the gaussians centers
        vertices = torch.tensor(pcd.vertices).float()
        faces = torch.tensor(pcd.faces).long()
        xyz = self.calc_xyz(alpha, vertices, faces).float().cuda()
        ## use vertex, face and scale to init self._scaling and self._rotation
        scale = torch.ones((pcd.points.shape[0], 1)).float() ## _scale means ro in paper, controlling the scale of gaussian
        scaling, rotation = self.calc_scaling_rot(alpha, scale, vertices, faces)
        scaling = scaling.float().cuda()
        rotation = rotation.float().cuda()

        self._xyz = nn.Parameter(xyz.requires_grad_(True))
        self._scaling = nn.Parameter(scaling.requires_grad_(True))
        self._rotation = nn.Parameter(rotation.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((pcd.points.shape[0]), device="cuda")


    def update_alpha(self, pcd: MeshPointCloud):
        """
        Function to control the alpha value.

        Alpha is the distance of the center of the gauss
         from the vertex of the triangle of the mesh.
        Thus, for each center of the gauss, 3 alphas
        are determined: alpha1+ alpha2+ alpha3.
        For a point to be in the center of the vertex,
        the alphas must meet the assumptions:
        alpha1 + alpha2 + alpha3 = 1
        and alpha1, alpha2, alpha3 >= 0

        #TODO
        check:
        # self.alpha = torch.relu(self._alpha)
        # self.alpha = self.alpha / self.alpha.sum(dim=-1, keepdim=True)

        """
        alpha_relu = torch.relu(pcd.alpha.float()) + 1e-8
        alpha = alpha_relu / alpha_relu.sum(dim=-1, keepdim=True)
        return alpha

    def calc_xyz(self, alpha, vertices, faces):
            """
            Calculate the 3D Gaussian center in the coordinates xyz.

            The alphas that are taken into account are the distances
            to the vertices and the coordinates of
            the triangles forming the mesh.

            Returns:
                None
            """
            xyz = torch.matmul(alpha, vertices[faces])
            print("Shape of xyz: ", xyz.shape)
            xyz = xyz.reshape(xyz.shape[0] * xyz.shape[1], 3)
            # print("re Shape of xyz: ", xyz.shape)
            return xyz

            
    def calc_scaling_rot(self, alpha, scale, vertices, faces, eps=1e-8):
        """
        approximate covariance matrix and calculate scaling/rotation tensors

        covariance matrix is [v0, v1, v2], where
        v0 is a normal vector to each face
        v1 is a vector from centroid of each face and 1st vertex
        v2 is obtained by orthogonal projection of a vector from centroid to 2nd vertex onto subspace spanned by v0 and v1

        Arguments:
        scale (torch.Tensor): float tensor with shape of [num_face * num_splat, 3]

        returns:
        rotations (torch.Tensor): float tensor with shape of [num_face * num_splat, 4]
        scaling (torch.Tensor): float tensor with shape of [num_face * num_splat, 3]
        """
        def dot(v, u):
            return (v * u).sum(dim=-1, keepdim=True)
        
        def proj(v, u):
            """
            projection of vector v onto subspace spanned by u

            vector u is assumed to be already normalized
            """
            coef = dot(v, u)
            return coef * u

        triangles = vertices[faces]
        normals = torch.linalg.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
            dim=1
        )
        v0 = normals / (torch.linalg.vector_norm(normals, dim=-1, keepdim=True) + eps)
        means = torch.mean(triangles, dim=1)
        v1 = triangles[:, 1] - means
        v1_norm = torch.linalg.vector_norm(v1, dim=-1, keepdim=True) + eps
        v1 = v1 / v1_norm
        v2_init = triangles[:, 2] - means
        v2 = v2_init - proj(v2_init, v0) - proj(v2_init, v1) # Gram-Schmidt
        v2 = v2 / (torch.linalg.vector_norm(v2, dim=-1, keepdim=True) + eps)

        s1 = v1_norm / 2.
        s2 = dot(v2_init, v2) / 2.
        s0 = eps * torch.ones_like(s1)
        scales = torch.concat((s0, s1, s2), dim=1).unsqueeze(dim=1)
        scales = scales.broadcast_to((*alpha.shape[:2], 3))
        scaling = torch.log(
            torch.nn.functional.relu(scale * scales.flatten(start_dim=0, end_dim=1)) + eps
        )

        rotation = torch.stack((v0, v1, v2), dim=1).unsqueeze(dim=1)
        rotation = rotation.broadcast_to((*alpha.shape[:2], 3, 3)).flatten(start_dim=0, end_dim=1)
        rotation = rotation.transpose(-2, -1)
        rotation = rot_to_quat_batch(rotation)

        return scaling, rotation

