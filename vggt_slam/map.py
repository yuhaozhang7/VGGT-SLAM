import os
import shutil
import numpy as np
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from vggt_slam.slam_utils import (
    decompose_camera,
    cosine_similarity,
    frame_id_to_timestamp_stem,
    timestamp_info_to_stem,
    timestamp_info_to_tum_string,
)

class GraphMap:
    def __init__(self):
        self.submaps = dict()
        self.rectifying_H_mats = []
        self.non_lc_submap_ids = []
    
    def get_num_submaps(self):
        return len(self.submaps)

    def add_submap(self, submap):
        submap_id = submap.get_id()
        self.submaps[submap_id] = submap
        if not submap.get_lc_status():
            self.non_lc_submap_ids.append(submap_id)
    
    def get_largest_key(self, ignore_loop_closure_submaps=False):
        """
        Get the largest key of the first node of any submap.
        Return: The largest key, or None if the dictionary is empty.
        """
        if len(self.submaps) == 0:
            return None
        if ignore_loop_closure_submaps:
            non_lc_keys = [key for key, submap in self.submaps.items() if not submap.get_lc_status()]
            return max(non_lc_keys)
        return max(self.submaps.keys())
    
    def get_submap(self, id):
        return self.submaps[id]

    def get_latest_submap(self, ignore_loop_closure_submaps=False):
        return self.get_submap(self.get_largest_key(ignore_loop_closure_submaps))

    def retrieve_best_semantic_frame(self, query_text_vector):
        overall_best_score = 0.0
        overall_best_submap_id = 0
        overall_best_frame_index = 0
        # search for best image to target image
        sorted_keys = sorted(self.submaps.keys())
        for index, submap_key in enumerate(sorted_keys):
            submap = self.submaps[submap_key]
            if submap.get_lc_status():
                continue
            submap_embeddings = submap.get_all_semantic_vectors()
            scores = []
            for index, embedding in enumerate(submap_embeddings):
                score = cosine_similarity(embedding, query_text_vector)
                scores.append(score)
            
            best_score_id = np.argmax(scores)
            best_score = scores[best_score_id]

            if best_score > overall_best_score:
                overall_best_score = best_score
                overall_best_submap_id = submap_key
                overall_best_frame_index = best_score_id

        return overall_best_score, overall_best_submap_id, overall_best_frame_index
    
    def retrieve_best_score_frame(self, query_vector, current_submap_id, ignore_last_submap=True):
        overall_best_score = 1000
        overall_best_submap_id = 0
        overall_best_frame_index = 0
        # search for best image to target image
        sorted_keys = sorted(self.submaps.keys())
        for index, submap_key in enumerate(sorted_keys):
            if submap_key == current_submap_id:
                continue

            if self.non_lc_submap_ids and ignore_last_submap and submap_key == self.non_lc_submap_ids[-1]:
                continue

            else:
                submap = self.submaps[submap_key]
                if submap.get_lc_status():
                    continue
                submap_embeddings = submap.get_all_retrieval_vectors()
                scores = []
                for index, embedding in enumerate(submap_embeddings):
                    score = torch.linalg.norm(embedding-query_vector)
                    # score = embedding @ query_vector.t()
                    scores.append(score.item())

                # for now assume we can only have at most one loop closure per submap
                
                best_score_id = np.argmin(scores)
                best_score = scores[best_score_id]

                if best_score < overall_best_score:
                    overall_best_score = best_score
                    overall_best_submap_id = submap_key
                    overall_best_frame_index = best_score_id

        return overall_best_score, overall_best_submap_id, overall_best_frame_index

    def get_frames_from_loops(self, loops):
        frames = []
        for detected_loop in loops:
            frames.append(self.submaps[detected_loop.detected_submap_id].get_frame_at_index(detected_loop.detected_submap_frame))
        return frames
    
    def get_submaps(self):
        return self.submaps.values()

    def ordered_submaps_by_key(self):
        for k in sorted(self.submaps):
            yield self.submaps[k]
    
    def get_all_homographies(self, graph):
        homographies = []
        for submap in self.ordered_submaps_by_key():
            for pose_num in range(len(submap.poses)):
                id = int(submap.get_id() + pose_num)
                homographies.append(graph.get_homography(id))
        return np.stack(homographies)

    def get_all_cam_matricies(self, graph, give_camera_mat):
        cam_mats = []
        for submap in self.ordered_submaps_by_key():
            if submap.get_lc_status():
                continue
            poses = submap.get_all_poses_world(graph, give_camera_mat=give_camera_mat)
            cam_mats.append(poses)
        return np.vstack(cam_mats)

    def write_poses_to_file(self, file_name, graph, give_camera_mat=False, kitti_format=False):
        all_poses = self.get_all_cam_matricies(give_camera_mat=True, graph=graph)
        with open(file_name, "w") as f:
            seen_timestamps = set()

            if self.rectifying_H_mats:
                assert len(self.rectifying_H_mats) == len(all_poses), "Number of rectifying mats and number of poses do not match"
                print("Using rectifying homographies when writing poses to file.")
            count = 0
            for submap_index, submap in enumerate(self.ordered_submaps_by_key()):
                if submap.get_lc_status():
                    continue
                frame_ids = submap.get_frame_ids()
                frame_timestamp_infos = submap.get_frame_timestamp_infos()
                print(frame_ids)
                for frame_index, frame_id in enumerate(frame_ids):
                    pose = all_poses[count]
                    timestamp_info = frame_timestamp_infos[frame_index]
                    timestamp_key = (
                        timestamp_info_to_stem(timestamp_info)
                        if timestamp_info is not None
                        else f"{float(frame_id):.9f}"
                    )
                    K, rotation_matrix, t, scale = decompose_camera(pose)
                    # print("Decomposed K:\n", K)
                    count += 1
                    if timestamp_key in seen_timestamps:
                        continue
                    seen_timestamps.add(timestamp_key)
                    x, y, z = t
                    if kitti_format:
                        pose_matrix = np.eye(4)
                        pose_matrix[:3, :3] = rotation_matrix
                        pose_matrix[:3, 3] = t
                        output = pose_matrix.flatten()[:-4]
                        output = np.array([float(frame_id), *output])
                        f.write(" ".join(f"{v:.8f}" for v in output) + "\n")
                    else:
                        quaternion = R.from_matrix(rotation_matrix).as_quat() # x, y, z, w
                        values = [x, y, z, *quaternion]
                        timestamp_str = (
                            timestamp_info_to_tum_string(timestamp_info)
                            if timestamp_info is not None
                            else f"{float(frame_id):.9f}"
                        )
                        f.write(
                            f"{timestamp_str} "
                            + " ".join(f"{v:.8f}" for v in values)
                            + "\n"
                        )

    def save_framewise_pointclouds(self, graph, file_name):
        os.makedirs(file_name, exist_ok=True)
        count = 0
        for submap in self.ordered_submaps_by_key():
            if submap.get_lc_status():
                continue
                count += len(submap.poses)
            pointclouds, frame_ids, conf_masks = submap.get_points_list_in_world_frame(graph)
            for frame_id, pointcloud, conf_masks in zip(frame_ids, pointclouds, conf_masks):
                # save pcd as numpy array
                np.savez(f"{file_name}/{frame_id}.npz", pointcloud=pointcloud, mask=conf_masks)
        assert count == len(self.rectifying_H_mats), "Number of rectifying mats and number of point maps do not match"
                

    def write_points_to_file(self, graph, file_name, include_loop_closure_submaps=True):
        pcd_all = []
        colors_all = []
        for submap in self.ordered_submaps_by_key():
            if not include_loop_closure_submaps and submap.get_lc_status():
                continue
            pcd = submap.get_points_in_world_frame(graph)
            pcd = pcd.reshape(-1, 3)
            pcd_all.append(pcd)
            colors_all.append(submap.get_points_colors())
        if not pcd_all:
            raise ValueError("No point clouds available to export.")
        pcd_all = np.concatenate(pcd_all, axis=0)
        colors_all = np.concatenate(colors_all, axis=0)
        if colors_all.max() > 1.0:
            colors_all = colors_all / 255.0
        pcd_all = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_all))
        pcd_all.colors = o3d.utility.Vector3dVector(colors_all)
        o3d.io.write_point_cloud(file_name, pcd_all)

    def write_global_outputs(self, graph, output_dir):
        if os.path.isdir(output_dir):
            for entry in os.listdir(output_dir):
                entry_path = os.path.join(output_dir, entry)
                if os.path.isdir(entry_path):
                    shutil.rmtree(entry_path)
                else:
                    os.remove(entry_path)
        else:
            os.makedirs(output_dir, exist_ok=True)
        self.write_points_to_file(
            graph,
            os.path.join(output_dir, "pts_global.pcd"),
            include_loop_closure_submaps=False,
        )
        self.write_poses_to_file(
            os.path.join(output_dir, "poses_tum.txt"),
            graph,
            kitti_format=False,
        )
        self.write_local_pointclouds(
            graph,
            os.path.join(output_dir, "pts_local"),
        )

    def write_local_pointclouds(self, graph, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        seen_timestamps = set()
        for submap in self.ordered_submaps_by_key():
            if submap.get_lc_status():
                continue

            pointclouds, frame_ids, conf_masks = submap.get_points_list_in_local_frame(graph)
            frame_timestamp_infos = submap.get_frame_timestamp_infos()
            for frame_id, timestamp_info, pointcloud, conf_mask, colors in zip(
                frame_ids,
                frame_timestamp_infos,
                pointclouds,
                conf_masks,
                submap.colors,
            ):
                points = pointcloud[conf_mask]
                point_colors = colors[conf_mask]
                if points.size == 0:
                    continue
                if point_colors.max() > 1.0:
                    point_colors = point_colors / 255.0

                pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
                pcd.colors = o3d.utility.Vector3dVector(point_colors)
                file_stem = (
                    timestamp_info_to_stem(timestamp_info)
                    if timestamp_info is not None
                    else frame_id_to_timestamp_stem(frame_id)
                )
                if file_stem in seen_timestamps:
                    continue
                seen_timestamps.add(file_stem)
                o3d.io.write_point_cloud(
                    os.path.join(output_dir, f"{file_stem}.pcd"),
                    pcd,
                )
