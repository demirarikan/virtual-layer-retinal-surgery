import cv2
import numpy as np
import open3d as o3d
import pyransac3d as pyrsc
from skimage.measure import LineModelND, ransac
from skimage.restoration import inpaint


class OctPointCloud:

    def __init__(self, seg_volume):
        self.seg_volume = seg_volume
        self.needle_points, self.ilm_points, self.rpe_points = (
            self.__find_first_occurrences(seg_volume)
        )
        self.cleaned_needle_points = None
        self.ilm_inpaint, self.rpe_inpaint = None, None

    def __find_first_occurrences(self, seg_volume, labels=[1, 2, 3]):
        """
        Finds the first occurrences of specified labels in a segmented volume.

        This method searches through a 3D segmented volume to find the first occurrence
        of each specified label along the first axis (depth axis). It returns the coordinates
        of these occurrences for each label.

        Args:
            seg_volume (np.ndarray): A 3D numpy array representing the segmented volume.
            labels (list, optional): A list of integer labels to search for in the segmented volume.
                                     Defaults to [1, 2, 3].

        Returns:
            tuple: A tuple containing three lists of coordinates. Each list corresponds to the
                   coordinates of the first occurrences of the respective label in the input list.
                   The coordinates are in the format [row, depth, column].
        """
        output_coordinates = []
        for value in labels:
            # Create a mask where the values are equal to the target value
            # all desired values will be set to 1 and all others to 0
            mask = (seg_volume == value).astype(
                np.uint8
            )  # uint8 because its slightly faster
            # set 0 values to 5 to avoid them being the minimum
            mask[mask == 0] = 5

            # Find the first occurrence by searching along the first axis (depth axis)
            # argmin returns first index if multiple minimum values are present
            # idx matrix non zero values will contain the first occurrence of the target value in each a-scan
            # idx has shape (seg_volume.shape[0], seg_volume.shape[2])
            indexes = np.argmin(mask, axis=1)
            rows, cols = np.nonzero(indexes)
            # row and column indexes are the same and the idx[row, col] will give the depth value
            coordinates = [[r, indexes[r, c], c] for r, c in zip(rows, cols)]
            output_coordinates.append(coordinates)

        return output_coordinates[0], output_coordinates[1], output_coordinates[2]

    def __needle_detection_scikit_ransac(self):
        np_needle_points = np.asarray(self.needle_points)
        _, inliers = ransac(
            np_needle_points,
            LineModelND,
            min_samples=2,
            residual_threshold=7,
            max_trials=250,
        )
        return np_needle_points[inliers]

    def find_needle_tip(self):
        """
        Finds the coordinates of the needle tip from the detected needle points.

        This method uses RANSAC to clean the detected needle points and then sorts
        and processes these points to find the lowest needle tip coordinates.

        Returns:
            numpy.ndarray: The coordinates of the needle tip.
        """
        cleaned_needle_points = self.__needle_detection_scikit_ransac()
        self.cleaned_needle_points = cleaned_needle_points

        sidx = np.lexsort(cleaned_needle_points[:, [1, 0]].T)
        idx = np.append(
            np.flatnonzero(
                cleaned_needle_points[1:, 0] > cleaned_needle_points[:-1, 0]
            ),
            cleaned_needle_points.shape[0] - 1,
        )
        lowest_needle_idx_coords = np.column_stack(
            (sidx[idx], cleaned_needle_points[sidx[idx]][:, 1])
        )
        lowest_needle_idx_coords = lowest_needle_idx_coords[
            np.argsort(lowest_needle_idx_coords[:, 1])
        ]
        try:
            needle_tip = lowest_needle_idx_coords[
                np.argwhere(
                    (
                        lowest_needle_idx_coords[:, 1][-1]
                        - lowest_needle_idx_coords[:, 1]
                    )
                    <= 25
                )
            ][0][0][0]
        except IndexError:
            needle_tip = lowest_needle_idx_coords[-1][0]
        needle_tip_coords = cleaned_needle_points[needle_tip]
        return needle_tip_coords

    def __get_depth_map(self, seg_index):
        """Create a depth map for the given segmentation index.

        Args:
            seg_index (int): 2 for ilm, 3 for rpe. Pixel value of the segmentation in the segmentation volume. 

        Raises:
            ValueError: Invalid segmentation index (anything other than 2 or 3).

        Returns:
            numpy.array: Array with the top down depth map of the given segmentation index.
        """
        z_dim, _, x_dim = self.seg_volume.shape
        depth_map = np.zeros((z_dim, x_dim))
        if seg_index == 2:
            layer_points = self.ilm_points
        elif seg_index == 3:
            layer_points = self.rpe_points
        else:
            raise ValueError("Invalid segmentation index")
        for point in layer_points:
            depth_map[point[0], point[2]] = point[1]
        return depth_map

    def __inpaint_layer(self, depth_map, debug=False):
        """Fill in the gaps in the depth map using OpenCV's inpaint function.

        Args:
            depth_map (numpy.array): Depth map of the layer to inpaint.
            debug (bool, optional): Show results in a new window. Defaults to False.

        Returns:
            numpy.array: Inpainted depth map.
        """
        depth_map_max = depth_map.max()
        # normalize
        depth_map = depth_map / depth_map_max
        # create inpainting mask
        inpainting_mask = np.where(depth_map == 0, 1, 0).astype(np.uint8)
        # inpaint
        inpaint_res = cv2.inpaint(
            depth_map.astype(np.float32), inpainting_mask, 3, cv2.INPAINT_NS
        )
        # inpaint_res = inpaint.inpaint_biharmonic(depth_map, inpainting_mask)
        # inpaint_res = set_outliers_to_mean_value(inpaint_res, threshold=0.5)
        if debug:

            def visualize_inpainting(depth_map, mask, inpaint_res):
                import matplotlib.pyplot as plt

                fig, axs = plt.subplots(3)
                axs[0].imshow(depth_map, cmap="gray")
                axs[0].set_title("Original depth map")
                axs[1].imshow(mask, cmap="gray")
                axs[1].set_title("Inpainting mask")
                axs[2].imshow(inpaint_res, cmap="gray")
                axs[2].set_title("Inpainting result")
                plt.show()

            visualize_inpainting(depth_map, inpainting_mask, inpaint_res)
        # denormalize
        inpaint_res = inpaint_res * depth_map_max
        return inpaint_res

    def inpaint_layers(self, debug=False):
        """Inpaint the ILM and RPE layers.

        Args:
            debug (bool, optional): Show the results in a new window. Defaults to False.

        Returns:
            numpy.array: Inpainted ILM and RPE layers.
        """
        ilm_depth_map = self.__get_depth_map(seg_index=2)
        rpe_depth_map = self.__get_depth_map(seg_index=3)

        inpainted_ilm = self.__inpaint_layer(ilm_depth_map, debug)
        inpainted_rpe = self.__inpaint_layer(rpe_depth_map, debug)

        self.ilm_inpaint = inpainted_ilm
        self.rpe_inpaint = inpainted_rpe

        return inpainted_ilm, inpainted_rpe

    # point cloud creation utilities
    def __create_needle_pcd(self, color=[1, 0, 0]):
        """Create open3D point cloud for the needle.

        Args:
            color (list, optional): Colors of the points. Defaults to [1, 0, 0].

        Returns:
            open3d.geometry.PointCloud: Point cloud object for the raw needle.
        """
        needle_pcd = o3d.geometry.PointCloud()
        needle_pcd.points = o3d.utility.Vector3dVector(self.needle_points)
        if color:
            needle_pcd.paint_uniform_color(color)
        return needle_pcd

    def __create_cleaned_needle_pcd(self, color=[1, 0, 0]):
        """Create open3D point cloud for the needle with the outliers removed.

        Args:
            color (list, optional): Colors of the points. Defaults to [1, 0, 0].

        Returns:
            open3d.geometry.PointCloud: Point cloud object for the cleaned needle.
        """
        cleaned_needle_pcd = o3d.geometry.PointCloud()
        cleaned_needle_pcd.points = o3d.utility.Vector3dVector(
            self.cleaned_needle_points
        )
        if color:
            cleaned_needle_pcd.paint_uniform_color(color)
        return cleaned_needle_pcd

    def __create_ilm_pcd(self, color=[0, 1, 0]):
        """Create open3d point cloud object for the ilm layer.

        Args:
            color (list, optional): Color of the points in the point cloud. Defaults to [0, 1, 0].

        Returns:
            open3d.geometry.PointCloud: Point cloud object for the ilm layer.
        """
        ilm_points = []
        for index_x in range(self.ilm_inpaint.shape[0]):
            for index_y in range(self.ilm_inpaint.shape[1]):
                ilm_points.append(
                    [index_x, self.ilm_inpaint[index_x, index_y], index_y]
                )

        ilm_pcd = o3d.geometry.PointCloud()
        ilm_pcd.points = o3d.utility.Vector3dVector(ilm_points)
        if color:
            ilm_pcd.paint_uniform_color(color)
        return ilm_pcd

    def __create_rpe_pcd(self, color=[0, 0, 1]):
        """Create open3d point cloud object for the RPE layer.

        Args:
            color (list, optional): Color of the points in the point cloud. Defaults to [0, 0, 1].

        Returns:
            open3d.geometry.PointCloud: Point cloud object for the RPE layer.
        """
        rpe_points = []
        for index_x in range(self.rpe_inpaint.shape[0]):
            for index_y in range(self.rpe_inpaint.shape[1]):
                rpe_points.append(
                    [index_x, self.rpe_inpaint[index_x, index_y], index_y]
                )

        rpe_pcd = o3d.geometry.PointCloud()
        rpe_pcd.points = o3d.utility.Vector3dVector(rpe_points)
        if color:
            rpe_pcd.paint_uniform_color(color)
        return rpe_pcd

    def __create_mesh_sphere(self, center, radius=3, color=[1.0, 0.0, 1.0]):
        """
        Create a mesh sphere with the given center, radius, and color.

        Parameters:
        - center (list): The center coordinates of the sphere in the form [slice, x, y].
        - radius (float): The radius of the sphere.
        - color (list): The color of the sphere in RGB format, with values ranging from 0 to 1.

        Returns:
        - mesh_sphere (o3d.geometry.TriangleMesh): The created mesh sphere.
        """

        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh_sphere.paint_uniform_color(color)

        your_transform = np.asarray(
            [
                [1.0, 0.0, 0.0, center[0]],
                [0.0, 1.0, 0.0, center[1]],
                [0.0, 0.0, 1.0, center[2]],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        mesh_sphere.transform(your_transform)
        return mesh_sphere

    def __create_mesh_cylinder(self, needle_tip_coords, radius=0.3, height=500):
        """Creates vertical cylinder mesh to visualize A-scan going through the needle tip.

        Args:
            needle_tip_coords (array): Position of the mesh.
            radius (float, optional): Radius of the cylinder. Defaults to 0.3.
            height (int, optional): Height of the cylinder. Defaults to 500.

        Returns:
            open3d.TriangleMesh: Mesh cylinder to visualize the A-scan.
        """
        ascan_cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius, height=height
        )
        transform = np.array(
            [
                [1, 0, 0, needle_tip_coords[0]],
                [0, 0, 1, needle_tip_coords[1]],
                [0, -1, 0, needle_tip_coords[2]],
                [0, 0, 0, 1],
            ]
        )
        ascan_cylinder.transform(transform)
        return ascan_cylinder

    def __create_virtual_layer(self, depth):
        """Creates the open3d point cloud object for the virtual target layer.

        Args:
            depth (float): Target depth of the virtual layer.

        Returns:
            open3d.geometry.PointCloud: Point cloud object for the virtual target layer.
        """
        ilm_pcd = self.__create_ilm_pcd()
        rpe_pcd = self.__create_rpe_pcd()
        virtual_pcd = o3d.geometry.PointCloud()
        virtual_points = []
        for ilm_point, rpe_point in zip(ilm_pcd.points, rpe_pcd.points):
            virtual_depth = depth * (rpe_point[1] - ilm_point[1]) + ilm_point[1]
            virtual_points.append([ilm_point[0], virtual_depth, ilm_point[2]])
        virtual_pcd.points = o3d.utility.Vector3dVector(virtual_points)
        virtual_pcd.paint_uniform_color([1, 0, 1])
        return virtual_pcd
            
    def create_point_cloud_components(
        self, needle_tip_coords, show_cleaned_needle=True
    ):
        """Create open3d components for the final point cloud visualization

        Args:
            needle_tip_coords (list): Needle tip coordinates that will be marked with a sphere and cylinder.
            show_cleaned_needle (bool, optional): Show needle with the outliers removed after RANSAC. Defaults to True.

        Returns:
            tuple: Tuple of needle tip coordinates, point clouds for the needle, ILM, and RPE layers, virtual target 
            layer and the meshes for needle tip sphere and A-scan cylinder.
        """
        if show_cleaned_needle:
            needle_pcd = self.__create_cleaned_needle_pcd()
        else:
            needle_pcd = self.__create_needle_pcd()

        ilm_pcd = self.__create_ilm_pcd()
        rpe_pcd = self.__create_rpe_pcd()

        needle_tip_sphere = self.__create_mesh_sphere(
            needle_tip_coords, radius=3, color=[1.0, 0.0, 1.0]
        )
        ascan_cylinder = self.__create_mesh_cylinder(
            needle_tip_coords, radius=0.3, height=500
        )

        virtual_layer = self.__create_virtual_layer(0.5)
        # self.create_before_pcd()
        # self.create_after_pcd()

        # o3d.visualization.draw_geometries([needle_pcd, ilm_pcd, rpe_pcd, needle_tip_sphere, ascan_cylinder, virtual_layer])

        return (
            # needle_tip_coords,
            needle_pcd,
            ilm_pcd,
            rpe_pcd,
            needle_tip_sphere,
            ascan_cylinder,
            virtual_layer
        )

    def save_pcd_visualization(
        self,
        geometries,
        focus,
        show_pcd=False,
        save_path="debug_log",
        save_name="point_cloud",
    ):
        """Generate and save a snapshot of the given geometries.

        Args:
            geometries (list): List of open3d geometries to visualize.
            focus (list): Camera focus point (usually the needle tip).
            show_pcd (bool, optional): Show the point cloud in a window. Defaults to False.
            save_path (str, optional): Path to save the snapshot. Defaults to "debug_log".
            save_name (str, optional): Name of the saved image file. Defaults to "point_cloud".
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        for geo in geometries:
            vis.add_geometry(geo)

        ctr = vis.get_view_control()

        # parameters = o3d.io.read_pinhole_camera_parameters("/home/demir/Desktop/jhu_project/oct-depth-control/ScreenCamera_2024-09-06-15-57-52.json")
        # ctr.convert_from_pinhole_camera_parameters(parameters)

        ctr.set_lookat(focus)
        ctr.set_up([0, -1, 0])
        ctr.set_front([1, 0, 0])
        ctr.set_zoom(0.2)

        vis.update_renderer()
        vis.capture_screen_image(f"{save_path}/{save_name}.png", True)
        if show_pcd:
            vis.run()
            vis.destroy_window()

    def draw_geometries(geos):
        o3d.visualization.draw_geometries(geos)

    def create_before_pcd(self):
        import os
        test_needle, test_ilm, test_rpe = o3d.geometry.PointCloud(), o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
        test_needle.points = o3d.utility.Vector3dVector(self.needle_points)
        test_ilm.points = o3d.utility.Vector3dVector(self.ilm_points)
        test_rpe.points = o3d.utility.Vector3dVector(self.rpe_points)
        test_needle.paint_uniform_color([1,0,0])
        test_ilm.paint_uniform_color([0,1,0])
        test_rpe.paint_uniform_color([0,0,1])


        o3d.visualization.draw_geometries([test_needle, test_ilm, test_rpe])
        

        # should check if name before.png is already taken in images folder
        # filename_base = "before/before"
        # filename_ext = ".ply"
        # filename = f"{filename_base}{filename_ext}"
        # counter = 1

        # # Increment the number in the filename if the file already exists
        # while os.path.exists(filename):
        #     filename = f"{filename_base}_{counter}{filename_ext}"
        #     counter += 1


        # o3d.io.write_point_cloud(filename, test_needle+test_ilm+test_rpe)

    def create_after_pcd(self):
        test_needle = o3d.geometry.PointCloud()
        test_needle.points = o3d.utility.Vector3dVector(self.cleaned_needle_points)
        test_ilm = self.__ilm_pcd()
        test_rpe = self.__rpe_pcd()
        test_needle.paint_uniform_color([1,0,0])
        o3d.io.write_point_cloud("after.ply", test_needle+test_ilm+test_rpe)
        # o3d.visualization.draw_geometries([test_needle, test_ilm, test_rpe])
