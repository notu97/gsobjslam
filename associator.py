import ultralytics.engine.results


class Associator:

    def __init__(self, configs: dict) -> None:

        self.configs = configs

    def associate(self, yolo_result: ultralytics.engine.results.Results, submaps: list) -> list:

        return yolo_result.boxes.id.int().cpu().tolist()
