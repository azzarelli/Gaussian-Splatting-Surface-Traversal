import os
from scene.gaussian_model import GaussianModel

class Scene:

    gaussians : GaussianModel

    def __init__(self, data):
        self.model_path = data

        # Load Gassian model
        self.gaussians = GaussianModel(data)

    def save(self, iteration, stage):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    