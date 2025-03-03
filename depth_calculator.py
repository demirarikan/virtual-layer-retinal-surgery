from oct_point_cloud import OctPointCloud


class DepthCalculator:
    def __init__(self, logger):
        self.logger = logger

    def calculate_depth(self, segmented_oct_volume, log_final_pcd=False):
        oct_pcd = OctPointCloud(seg_volume=segmented_oct_volume)
        needle_tip_coords = oct_pcd.find_needle_tip()
        inpainted_ilm, inpainted_rpe = oct_pcd.inpaint_layers(debug=False)
        _, _, current_depth_relative, layer_depths = self.calculate_needle_tip_depth(
            needle_tip_coords, inpainted_ilm, inpainted_rpe
        )
        components = None
        if log_final_pcd:
            # self.logger.log_pcd(
            #     oct_pcd.create_point_cloud_components(needle_tip_coords)
            # )
            components = oct_pcd.create_point_cloud_components(needle_tip_coords)
            components = layer_depths + components
            
        return current_depth_relative, components

    def calculate_needle_tip_depth(
        self, needle_tip_coords, inpainted_ilm, inpainted_rpe
    ):
        needle_tip_depth = needle_tip_coords[1]
        ilm_depth = inpainted_ilm[needle_tip_coords[0], needle_tip_coords[2]]
        rpe_depth = inpainted_rpe[needle_tip_coords[0], needle_tip_coords[2]]
        ilm_rpe_distance = rpe_depth - ilm_depth
        needle_tip_depth_relative = needle_tip_depth - ilm_depth
        needle_tip_depth_relative_percentage = (
            needle_tip_depth_relative / ilm_rpe_distance
        )
        return (
            needle_tip_depth,
            needle_tip_depth_relative,
            needle_tip_depth_relative_percentage,
            (ilm_depth, rpe_depth),
        )