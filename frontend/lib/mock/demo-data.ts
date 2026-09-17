import type { Experiment } from "../types";

export const experiments: Experiment[] = [
  {
    id: "EXP001",
    label: "100-epoch baseline",
    model: "YOLOv8n-Seg",
    imageSize: 512,
    batch: 4,
    epochs: 100,
    learningRate: "0.01 / optimizer auto",
    augmentation: "Ultralytics defaults",
    weightDecay: "0.0005",
    status: "Validation + held-out test",
    f1: 0.712,
    precision: 0.744,
    recall: 0.683,
    map50: 0.696,
    map5095: 0.236,
    bestEpoch: null,
  },
  {
    id: "EXP002",
    label: "25-epoch hyperparameter screen",
    model: "YOLOv8n-Seg",
    imageSize: 640,
    batch: 24,
    epochs: 25,
    learningRate: "0.001",
    augmentation: "Medium",
    weightDecay: "0.0005",
    status: "Validation only",
    f1: 0.719758,
    precision: 0.776583,
    recall: 0.670683,
    map50: 0.65467,
    map5095: 0.2161,
    bestEpoch: 22,
  },
];

export const datasetStats = {
  total: 4028,
  train: 3717,
  val: 199,
  test: 112,
  excluded: 1,
  warnings: 5,
};
