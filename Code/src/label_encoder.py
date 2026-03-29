from typing import Dict, List

import numpy as np


OBJECT_CLASSES = ["building", "road", "vegetation"]
ACTION_CLASSES = ["add", "remove", "replace", "rebuild"]
LOCATION_CLASSES = [
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
    "center",
    "top",
    "bottom",
    "left",
    "right",
    "corner",
]


class LabelEncoder:
    """集中管理 LEVIR-MCI 的标签映射与编码逻辑。"""

    def __init__(self):
        self.object_classes = OBJECT_CLASSES
        self.action_classes = ACTION_CLASSES
        self.location_classes = LOCATION_CLASSES

        self.object_to_idx = {name: i for i, name in enumerate(self.object_classes)}
        self.action_to_idx = {name: i for i, name in enumerate(self.action_classes)}
        self.location_to_idx = {name: i for i, name in enumerate(self.location_classes)}

    @staticmethod
    def _to_multihot(labels: List[str], mapping: Dict[str, int], n_classes: int) -> np.ndarray:
        vec = np.zeros(n_classes, dtype=np.float32)
        for label in labels:
            if label not in mapping:
                raise ValueError(f"Unknown label '{label}'. Valid labels: {list(mapping.keys())}")
            vec[mapping[label]] = 1.0
        return vec

    def encode_change(self, changeflag: int) -> float:
        if changeflag not in (0, 1):
            raise ValueError(f"changeflag must be 0/1, got: {changeflag}")
        return float(changeflag)

    def encode_sample(self, sample: Dict) -> Dict[str, np.ndarray]:
        return {
            "change_label": np.array(self.encode_change(sample["changeflag"]), dtype=np.float32),
            "object_labels": self._to_multihot(sample.get("object_labels", []), self.object_to_idx, len(self.object_classes)),
            "action_labels": self._to_multihot(sample.get("action_labels", []), self.action_to_idx, len(self.action_classes)),
            "location_labels": self._to_multihot(sample.get("location_labels", []), self.location_to_idx, len(self.location_classes)),
        }

    @staticmethod
    def decode_multihot(vec: np.ndarray, class_names: List[str], threshold: float = 0.5) -> List[str]:
        if vec.ndim != 1:
            raise ValueError(f"decode_multihot expects 1D vector, got shape: {vec.shape}")
        return [class_names[i] for i, v in enumerate(vec) if float(v) >= threshold]
