import ultralytics.engine.results


class Associator:

    def __init__(self, configs: dict) -> None:

        self.configs = configs

    def associate(self, yolo_result: ultralytics.engine.results.Results, object_idx: int,
                  submaps: list) -> int:

        # @TODO: implement data association
        return -1
