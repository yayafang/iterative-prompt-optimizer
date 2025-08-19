from typing import Dict


class RunMetrics:
    def __init__(self, experiment_name: str, run_name: str):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.epoch_metrics: Dict[int, Dict] = {}
        self.overall_metrics: Dict = {
            "best_accuracy": 0,
            "best_epoch": 0,
            "best_update": 0,
        }

    def update(self, epoch: int, batch: int, accuracy: float):
        if epoch not in self.epoch_metrics:
            self.epoch_metrics[epoch] = {"updates": {}, "accuracy": 0}

        self.epoch_metrics[epoch]["updates"][batch] = accuracy
        self.epoch_metrics[epoch]["accuracy"] = sum(
            self.epoch_metrics[epoch]["updates"].values()
        ) / len(self.epoch_metrics[epoch]["updates"])

        if (
            self.epoch_metrics[epoch]["accuracy"]
            > self.overall_metrics["best_accuracy"]
        ):
            self.overall_metrics.update(
                {
                    "best_accuracy": self.epoch_metrics[epoch]["accuracy"],
                    "best_epoch": epoch,
                    "best_update": max(self.epoch_metrics[epoch]["updates"].keys()),
                }
            )

    def to_dict(self) -> Dict:
        return {
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "epoch_metrics": self.epoch_metrics,
            "overall_metrics": self.overall_metrics,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "RunMetrics":
        instance = cls(data["experiment_name"], data["run_name"])
        instance.epoch_metrics = data["epoch_metrics"]
        instance.overall_metrics = data["overall_metrics"]
        return instance
