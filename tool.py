from typing import Dict, Optional, Tuple, Type

import numpy as np
import torch
from langchain_core.callbacks import (AsyncCallbackManagerForToolRun,
                                      CallbackManagerForToolRun)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from utils import test_single_volume

CASE_NAME = 'case0022'


def _load_model():
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = 9
    config_vit.n_skip = 3
    config_vit.patches.grid = (int(224 / 16), int(224 / 16))
    model = ViT_seg(config_vit, img_size=224,
                    num_classes=config_vit.n_classes).cuda()
    model.load_from(weights=np.load(config_vit.pretrained_path))

    model.load_state_dict(torch.load(
        "./model/TU_Synapse224/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/epoch_149.pth", weights_only=True))
    return model


class SamplesInput(BaseModel):
    """Input for chest X-ray analysis tools. Only supports JPG or PNG images."""

    image_path: str = Field(
        ..., description="Path to the radiology image file, only supports JPG or PNG images"
    )


class OrganSegmentTool(BaseTool):

    name: str = "organ_segment_tool"
    description: str = (
        "aorta"
        "gallbladder"
        "left kidney"
        "right kidney"
        "liver"
        "pancreas"
        "spleen"
        "stomach"
    )
    args_schema: Type[BaseModel] = SamplesInput
    model: ViT_seg = None
    device: Optional[str] = "cuda"

    def __init__(self, model_name: str = "", device: Optional[str] = "cuda"):
        super().__init__()
        self.model = _load_model()
        self.model.eval()
        self.device = torch.device(device if device else "cuda")
        self.model = self.model.to(self.device)

    def _process_image(self, image_path: str) -> torch.Tensor:
        image = np.load(image_path)
        label = np.load(image_path.replace('image', 'label'))
        sample = {'image': image, 'label': label}
        sample['case_name'] = CASE_NAME
        return sample

    def _run(
        self,
        image_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, float], Dict]:
        try:
            sample = self._process_image(image_path)
            image, label = sample['image'], sample['label']
            metric_list = 0.0

            metric_i, preds = test_single_volume(image, label, self.model, classes=9, patch_size=[224, 224],
                                                 test_save_path='./predictions', case=sample['case_name'], z_spacing=1)
            metric_list += np.array(metric_i)
            performance = np.mean(metric_list, axis=0)[0]
            mean_hd95 = np.mean(metric_list, axis=0)[1]

            default_pathologies = ['Image Type']
            output = dict(zip(default_pathologies, preds))
            metadata = {
                "image_path": image_path,
                "analysis_status": "completed",
                "note": F"case name: {CASE_NAME}, dice: {performance}, hd95: {mean_hd95}",
            }

            return output, metadata
        except Exception as e:
            return {"error": str(e)}, {
                "image_path": image_path,
                "analysis_status": "failed",
            }

    async def _arun(
        self,
        image_path: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, float], Dict]:
        return self._run(image_path)


if __name__ == "__main__":
    image_path = "./samples/case0022_image.npy"
    tool = OrganSegmentTool()
    output, metadata = tool(image_path)
    print(output)
    print(metadata)
