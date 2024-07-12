"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""
import warnings

from carvekit.api.interface import Interface
from ldm.fba_matting import FBAMatting
from carvekit.ml.wrap.tracer_b7 import TracerUniversalB7
from carvekit.ml.wrap.u2net import U2NET
from carvekit.pipelines.postprocessing import MattingMethod
from carvekit.trimap.generator import TrimapGenerator


class HiInterface(Interface):
    def __init__(
        self,
        object_type: str = "object",
        batch_size_seg=2,
        batch_size_matting=1,
        device="cpu",
        seg_mask_size=640,
        matting_mask_size=2048,
        trimap_prob_threshold=231,
        trimap_dilation=30,
        trimap_erosion_iters=5,
        fp16=False,
    ):
        """
        Initializes High Level interface.

        Args:
            object_type: Interest object type. Can be "object" or "hairs-like".
            matting_mask_size:  The size of the input image for the matting neural network.
            seg_mask_size: The size of the input image for the segmentation neural network.
            batch_size_seg: Number of images processed per one segmentation neural network call.
            batch_size_matting: Number of images processed per one matting neural network call.
            device: Processing device
            fp16: Use half precision. Reduce memory usage and increase speed. Experimental support
            trimap_prob_threshold: Probability threshold at which the prob_filter and prob_as_unknown_area operations will be applied
            trimap_dilation: The size of the offset radius from the object mask in pixels when forming an unknown area
            trimap_erosion_iters: The number of iterations of erosion that the object's mask will be subjected to before forming an unknown area

        Notes:
            1. Changing seg_mask_size may cause an out-of-memory error if the value is too large, and it may also
            result in reduced precision. I do not recommend changing this value. You can change matting_mask_size in
            range from (1024 to 4096) to improve object edge refining quality, but it will cause extra large RAM and
            video memory consume. Also, you can change batch size to accelerate background removal, but it also causes
            extra large video memory consume, if value is too big.

            2. Changing trimap_prob_threshold, trimap_kernel_size, trimap_erosion_iters may improve object edge
            refining quality,
        """
        if object_type == "object":
            self.u2net = TracerUniversalB7(
                device=device,
                batch_size=batch_size_seg,
                input_image_size=seg_mask_size,
                fp16=fp16,
                model_path = './ptms/tracer_b7/tracer_b7.pth'
            )
        elif object_type == "hairs-like":
            self.u2net = U2NET(
                device=device,
                batch_size=batch_size_seg,
                input_image_size=seg_mask_size,
                fp16=fp16,
            )
        else:
            warnings.warn(
                f"Unknown object type: {object_type}. Using default object type: object"
            )
            self.u2net = TracerUniversalB7(
                device=device,
                batch_size=batch_size_seg,
                input_image_size=seg_mask_size,
                fp16=fp16,
                model_path = './ptms/tracer_b7/tracer_b7.pth'
            )

        self.fba = FBAMatting(
            batch_size=batch_size_matting,
            device=device,
            input_tensor_size=matting_mask_size,
            fp16=fp16,
        )
        self.trimap_generator = TrimapGenerator(
            prob_threshold=trimap_prob_threshold,
            kernel_size=trimap_dilation,
            erosion_iters=trimap_erosion_iters,
        )
        super(HiInterface, self).__init__(
            pre_pipe=None,
            seg_pipe=self.u2net,
            post_pipe=MattingMethod(
                matting_module=self.fba,
                trimap_generator=self.trimap_generator,
                device=device,
            ),
            device=device,
        )