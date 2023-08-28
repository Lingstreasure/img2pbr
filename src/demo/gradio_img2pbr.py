from typing import Tuple

import gradio as gr
import rootutils
from PIL import Image

from src.demo.inference_img2pbr import high_res_img2pbr, img2pbr

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def wrapper(img: Image.Image) -> Tuple[Image.Image]:
    """A wrapper function of gradio."""
    albedo, normal, rough = high_res_img2pbr(img)
    return albedo, normal, rough


if __name__ == "__main__":
    # p1 = "/home/d5/dw/sdmat/outputs/tiling_exp/all_pbr_0426_notiling/samples/out_0_wood with stratified pattern.png"
    # p2 = "/home/d5/dw/sdmat/outputs/tiling_exp/all_pbr_0426_notiling/samples/out_2_pebbles.png"
    # p3 = "/home/d5/dw/sdmat/outputs/all_pbr_0530_exp4/samples/out_1_marble.png"
    # examples = [
    #     p1,
    #     p2,
    # ]

    input1 = gr.inputs.Image(type="pil")

    # preprocess
    input1

    gr.Interface(
        wrapper,
        inputs=[input1],
        # outputs = ["image", "image", "image"],
        outputs=[gr.Gallery()],
        title="AI Material Image to PBR",
        description="output albedo normal roughness",
        #  examples=examples
    ).launch(server_name="1.0.2.48", share=True)
