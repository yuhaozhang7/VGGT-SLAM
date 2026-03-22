import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def _import_depth_anything_3():
    try:
        from depth_anything_3.api import DepthAnything3
        return DepthAnything3
    except ModuleNotFoundError:
        da3_src_dir = Path(__file__).resolve().parents[2] / "Depth-Anything-3" / "src"
        if da3_src_dir.is_dir():
            sys.path.insert(0, str(da3_src_dir))
            from depth_anything_3.api import DepthAnything3
            return DepthAnything3
        raise


class DepthAnything3OnlyWrapper(nn.Module):
    def __init__(
        self,
        device,
        da3_model_name="depth-anything/DA3NESTED-GIANT-LARGE",
        process_res=504,
        process_res_method="upper_bound_resize",
    ):
        super().__init__()
        self.device_name = device
        self.process_res = process_res
        self.process_res_method = process_res_method
        self.uses_da3_only = True

        DepthAnything3 = _import_depth_anything_3()
        self.da3 = DepthAnything3.from_pretrained(da3_model_name).to(device=device)
        self.da3.eval()

    def predict_from_paths(self, image_paths):
        prediction = self.da3.inference(
            image=image_paths,
            process_res=self.process_res,
            process_res_method=self.process_res_method,
        )
        depth = prediction.depth
        depth_conf = prediction.conf
        if depth_conf is None:
            depth_conf = np.ones_like(depth, dtype=np.float32)

        extrinsics = prediction.extrinsics
        if extrinsics is None:
            raise ValueError("DA3 did not return extrinsics in DA-only mode.")
        processed_images = prediction.processed_images
        if processed_images is None:
            raise ValueError("DA3 did not return processed images in DA-only mode.")
        processed_images = torch.from_numpy(processed_images).permute(0, 3, 1, 2).float() / 255.0

        return {
            "images": processed_images,
            "depth": depth[..., None],
            "depth_conf": depth_conf,
            "extrinsic": extrinsics,
            "intrinsic": prediction.intrinsics,
        }
