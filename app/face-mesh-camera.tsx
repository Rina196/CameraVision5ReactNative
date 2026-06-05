import { CameraRoll } from "@react-native-camera-roll/camera-roll";
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { StyleSheet, Text, TouchableOpacity, View } from "react-native";
import {
  ScalarType,
  useExecutorchModule,
  type TensorPtr,
} from "react-native-executorch";
import { loadImage, type Image } from "react-native-nitro-image";
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  usePhotoOutput,
} from "react-native-vision-camera";

type FaceBox = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

type FaceKeypoint = {
  x: number;
  y: number;
  score?: number;
};

type Detection = {
  label: string;
  score: number;
  bbox: FaceBox;
  keypoints: FaceKeypoint[];
  embedding: Float32Array | null;
  similarity: number | null;
  recognizedName: string | null;
};

type TrackedDetection = {
  id: string;
  detection: Detection;
};

type SnapshotInfo = {
  width: number;
  height: number;
};

type RawPixelFormat = ReturnType<Image["toRawPixelData"]>["pixelFormat"];

type ParsedFaceOutputs = {
  boxes: Float32Array;
  boxesShape: number[];
  scores: Float32Array;
  scoresShape: number[];
  boxEncoding: "xyxy" | "dfl" | "xywh";
  fusedScores: Float32Array | null;
  fusedScoresShape: number[] | null;
  embeddings: Float32Array | null;
  embeddingsShape: number[] | null;
  keypoints: Float32Array | null;
  keypointsShape: number[] | null;
};

type ReferenceFace = {
  name: string;
  embedding: Float32Array;
};

const MODEL_WIDTH = 640;
const MODEL_HEIGHT = 640;
const CONF_THRESHOLD = 0.2;
const IOU_THRESHOLD = 0.45;
const RECOGNITION_THRESHOLD = 0.45;
const REG_MAX = 16;
const STRIDES = [8, 16, 32] as const;
const MIN_FACE_SIZE = 56;
const MIN_FACE_AREA_RATIO = 0.006;
const MAX_RENDERED_FACES = 1;
const MIN_FACE_WIDTH_RATIO = 0.16;
const MIN_FACE_HEIGHT_RATIO = 0.22;

// eslint-disable-next-line @typescript-eslint/no-require-imports
const YOLO_FACE_MODEL = require("../assets/models/yolov8n-face.pte");

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function sigmoid(value: number) {
  return 1 / (1 + Math.exp(-value));
}

function softmax(values: number[]) {
  const maxValue = Math.max(...values);
  const exps = values.map((value) => Math.exp(value - maxValue));
  const sum = exps.reduce((acc, value) => acc + value, 0);
  return exps.map((value) => value / sum);
}

function iou(a: FaceBox, b: FaceBox) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);

  if (inter <= 0) {
    return 0;
  }

  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);

  return inter / (areaA + areaB - inter);
}

function nms(detections: Detection[], threshold: number) {
  const sorted = [...detections].sort((a, b) => b.score - a.score);
  const keep: Detection[] = [];

  while (sorted.length > 0) {
    const best = sorted.shift()!;
    keep.push(best);

    for (let i = sorted.length - 1; i >= 0; i -= 1) {
      if (iou(best.bbox, sorted[i].bbox) > threshold) {
        sorted.splice(i, 1);
      }
    }
  }

  return keep;
}

function getBoxWidth(box: FaceBox) {
  return Math.max(0, box.x2 - box.x1);
}

function getBoxHeight(box: FaceBox) {
  return Math.max(0, box.y2 - box.y1);
}

function getBoxArea(box: FaceBox) {
  return getBoxWidth(box) * getBoxHeight(box);
}

function distance(a: FaceKeypoint, b: FaceKeypoint) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function createBoxFromCenter(
  centerX: number,
  centerY: number,
  width: number,
  height: number,
  snapshot: SnapshotInfo,
) {
  return {
    x1: clamp(centerX - width / 2, 0, snapshot.width),
    y1: clamp(centerY - height / 2, 0, snapshot.height),
    x2: clamp(centerX + width / 2, 0, snapshot.width),
    y2: clamp(centerY + height / 2, 0, snapshot.height),
  };
}

function createAdjustedFaceBox(
  centerX: number,
  centerY: number,
  width: number,
  height: number,
  snapshot: SnapshotInfo,
) {
  const shiftedCenterX = centerX + width * 0.08;
  const shiftedCenterY = centerY + height * 0.05;
  const adjustedWidth = width * 0.8;
  const adjustedHeight = height * 0.78;

  return createBoxFromCenter(
    shiftedCenterX,
    shiftedCenterY,
    adjustedWidth,
    adjustedHeight,
    snapshot,
  );
}

function deriveFaceBoxFromKeypoints(
  keypoints: FaceKeypoint[],
  fallbackBox: FaceBox,
  snapshot: SnapshotInfo,
) {
  const confident = keypoints.filter((point) => (point.score ?? 1) >= 0.15);

  const fallbackCenterX = (fallbackBox.x1 + fallbackBox.x2) / 2;
  const fallbackCenterY = (fallbackBox.y1 + fallbackBox.y2) / 2;
  const fallbackWidth = Math.max(1, getBoxWidth(fallbackBox));
  const fallbackHeight = Math.max(1, getBoxHeight(fallbackBox));
  const minimumWidth = Math.max(
    MIN_FACE_SIZE,
    snapshot.width * MIN_FACE_WIDTH_RATIO,
  );
  const minimumHeight = Math.max(
    MIN_FACE_SIZE,
    snapshot.height * MIN_FACE_HEIGHT_RATIO,
  );

  if (confident.length < 3) {
    return createAdjustedFaceBox(
      fallbackCenterX,
      fallbackCenterY,
      Math.max(fallbackWidth * 14, minimumWidth),
      Math.max(fallbackHeight * 18, minimumHeight),
      snapshot,
    );
  }

  const minX = Math.min(...confident.map((point) => point.x));
  const maxX = Math.max(...confident.map((point) => point.x));
  const minY = Math.min(...confident.map((point) => point.y));
  const maxY = Math.max(...confident.map((point) => point.y));
  const spreadX = Math.max(1, maxX - minX);
  const spreadY = Math.max(1, maxY - minY);

  if (keypoints.length >= 5) {
    const leftEye = keypoints[0];
    const rightEye = keypoints[1];
    const nose = keypoints[2];
    const leftMouth = keypoints[3];
    const rightMouth = keypoints[4];
    const eyeCenterX = (leftEye.x + rightEye.x) / 2;
    const eyeCenterY = (leftEye.y + rightEye.y) / 2;
    const mouthCenterX = (leftMouth.x + rightMouth.x) / 2;
    const mouthCenterY = (leftMouth.y + rightMouth.y) / 2;
    const eyeDistance = Math.max(1, distance(leftEye, rightEye));
    const eyeToMouth = Math.max(
      1,
      Math.hypot(mouthCenterX - eyeCenterX, mouthCenterY - eyeCenterY),
    );
    const faceMidX = (eyeCenterX + mouthCenterX) / 2;
    const faceMidY = eyeCenterY * 0.4 + mouthCenterY * 0.6;

    console.log("deriveFaceBoxFromKeypoints", {
      eyeCenterX,
      eyeCenterY,
      mouthCenterX,
      mouthCenterY,
      eyeDistance,
      eyeToMouth,
      faceMidX,
      faceMidY,
    });
    const width = Math.max(
      spreadX * 2.0,
      eyeDistance * 2.15,
      fallbackWidth * 8,
      minimumWidth,
    );
    const height = Math.max(
      spreadY * 2.25,
      eyeToMouth * 2.75,
      fallbackHeight * 10,
      minimumHeight,
    );
    const centerX = faceMidX;
    const centerY = faceMidY + eyeToMouth * 0.08;

    return createAdjustedFaceBox(centerX, centerY, width, height, snapshot);
  }

  return createAdjustedFaceBox(
    (minX + maxX) / 2,
    (minY + maxY) / 2 + Math.max(spreadY * 0.18, minimumHeight * 0.06),
    Math.max(spreadX * 2.0, fallbackWidth * 8, minimumWidth),
    Math.max(spreadY * 2.35, fallbackHeight * 10, minimumHeight),
    snapshot,
  );
}

function isPlausibleFaceBox(box: FaceBox, snapshot: SnapshotInfo) {
  const width = getBoxWidth(box);
  const height = getBoxHeight(box);
  const areaRatio =
    getBoxArea(box) / Math.max(1, snapshot.width * snapshot.height);

  return (
    width >= MIN_FACE_SIZE &&
    height >= MIN_FACE_SIZE &&
    areaRatio >= MIN_FACE_AREA_RATIO &&
    width / Math.max(1, height) >= 0.5 &&
    width / Math.max(1, height) <= 1.8
  );
}

function getFacePriority(detection: Detection, snapshot: SnapshotInfo) {
  const width = getBoxWidth(detection.bbox);
  const height = getBoxHeight(detection.bbox);
  const areaRatio =
    getBoxArea(detection.bbox) / Math.max(1, snapshot.width * snapshot.height);
  const centerX = (detection.bbox.x1 + detection.bbox.x2) / 2;
  const centerY = (detection.bbox.y1 + detection.bbox.y2) / 2;
  const dx = centerX / snapshot.width - 0.5;
  const dy = centerY / snapshot.height - 0.5;
  const centerPenalty = Math.sqrt(dx * dx + dy * dy);
  const aspectRatio = width / Math.max(1, height);
  const aspectPenalty = Math.abs(1 - aspectRatio);

  return (
    detection.score * 2 +
    areaRatio * 12 -
    centerPenalty * 1.5 -
    aspectPenalty * 0.75
  );
}

function finalizeDetections(detections: Detection[], snapshot: SnapshotInfo) {
  const ranked = nms(detections, IOU_THRESHOLD);
  const plausible = ranked
    .filter((detection) => isPlausibleFaceBox(detection.bbox, snapshot))
    .sort(
      (a, b) => getFacePriority(b, snapshot) - getFacePriority(a, snapshot),
    );

  if (plausible.length > 0) {
    return plausible.slice(0, MAX_RENDERED_FACES);
  }

  return ranked
    .sort((a, b) => getFacePriority(b, snapshot) - getFacePriority(a, snapshot))
    .slice(0, Math.min(MAX_RENDERED_FACES, ranked.length));
}

type TensorVariant = {
  id: string;
  channelOrder: "rgb" | "bgr";
  normalize: boolean;
};

function imageToFloatChw(
  rawBuffer: ArrayBuffer,
  pixelFormat: RawPixelFormat,
  variant: TensorVariant,
) {
  const pixels = new Uint8Array(rawBuffer);
  const pixelCount = MODEL_WIDTH * MODEL_HEIGHT;
  const chw = new Float32Array(pixelCount * 3);
  const is3Channel = pixelFormat === "RGB" || pixelFormat === "BGR";

  for (let i = 0; i < pixelCount; i += 1) {
    const offset = i * (is3Channel ? 3 : 4);
    let r = 0;
    let g = 0;
    let b = 0;

    switch (pixelFormat) {
      case "RGBA":
      case "RGBX":
        r = pixels[offset];
        g = pixels[offset + 1];
        b = pixels[offset + 2];
        break;
      case "BGRA":
      case "BGRX":
        b = pixels[offset];
        g = pixels[offset + 1];
        r = pixels[offset + 2];
        break;
      case "RGB":
        r = pixels[offset];
        g = pixels[offset + 1];
        b = pixels[offset + 2];
        break;
      case "BGR":
        b = pixels[offset];
        g = pixels[offset + 1];
        r = pixels[offset + 2];
        break;
      default:
        throw new Error(`Unsupported pixel format: ${pixelFormat}`);
    }

    const scale = variant.normalize ? 1 / 255 : 1;
    const first = variant.channelOrder === "rgb" ? r : b;
    const second = g;
    const third = variant.channelOrder === "rgb" ? b : r;

    chw[i] = first * scale;
    chw[pixelCount + i] = second * scale;
    chw[pixelCount * 2 + i] = third * scale;
  }

  return chw;
}

function toFloatArray(tensor: TensorPtr) {
  if (tensor.dataPtr instanceof Float32Array) {
    return tensor.dataPtr;
  }

  return new Float32Array(tensor.dataPtr as ArrayBufferLike);
}

function getLastDim(shape: number[]) {
  return shape[shape.length - 1] ?? 0;
}

function getSecondLastDim(shape: number[]) {
  return shape[shape.length - 2] ?? 0;
}

function getDetectionCount(shape: number[]) {
  if (shape.length === 0) {
    return 0;
  }

  const lastDim = getLastDim(shape);
  const secondLastDim = getSecondLastDim(shape);

  if (
    lastDim === 4 ||
    lastDim === 10 ||
    lastDim === 15 ||
    lastDim === 20 ||
    lastDim >= 16
  ) {
    return secondLastDim || shape[0] || 0;
  }

  return lastDim || shape[0] || 0;
}

function isLikelyBoxes(shape: number[]) {
  return getLastDim(shape) === 4 || getSecondLastDim(shape) === 4;
}

function isLikelyKeypoints(shape: number[]) {
  const lastDim = getLastDim(shape);
  const secondLastDim = getSecondLastDim(shape);

  return lastDim === 10 || lastDim === 15 || secondLastDim === 5;
}

function isLikelyEmbeddings(shape: number[]) {
  const lastDim = getLastDim(shape);
  const secondLastDim = getSecondLastDim(shape);

  return (
    (lastDim >= 16 && lastDim <= 1024 && lastDim !== 64) ||
    (secondLastDim >= 16 && secondLastDim <= 1024 && secondLastDim !== 64)
  );
}

function inferScoreAtIndex(
  values: Float32Array,
  shape: number[],
  index: number,
) {
  if (shape.length === 0) {
    return values[index] ?? 0;
  }

  const lastDim = getLastDim(shape);
  const secondLastDim = getSecondLastDim(shape);

  if (lastDim === 1 && secondLastDim > 0) {
    return values[index] ?? 0;
  }

  if (shape.length >= 2 && lastDim === 2) {
    const scoreA = values[index * 2] ?? 0;
    const scoreB = values[index * 2 + 1] ?? 0;
    return Math.max(scoreA, scoreB);
  }

  if (shape.length >= 2 && secondLastDim === 2 && lastDim > 2) {
    return values[index] ?? 0;
  }

  return values[index] ?? 0;
}

function getCombinedTensorValue(
  values: Float32Array,
  channels: number,
  anchors: number,
  channel: number,
  anchorIndex: number,
) {
  return values[channel * anchors + anchorIndex] ?? 0;
}

function sliceEmbedding(
  values: Float32Array | null,
  shape: number[] | null,
  index: number,
) {
  if (!values || !shape) {
    return null;
  }

  const width = getLastDim(shape);

  if (!width || shape.length < 2) {
    return null;
  }

  const start = index * width;
  const end = start + width;

  if (end > values.length) {
    return null;
  }

  return values.slice(start, end);
}

function sliceKeypoints(
  values: Float32Array | null,
  shape: number[] | null,
  index: number,
  snapshot: SnapshotInfo,
) {
  if (!values || !shape) {
    return [];
  }

  const lastDim = getLastDim(shape);
  const start = index * lastDim;
  const end = start + lastDim;

  if (lastDim < 10 || end > values.length) {
    return [];
  }

  const raw = values.slice(start, end);
  const normalized = raw.every((value, rawIndex) => {
    if (rawIndex % 3 === 2) {
      return value >= 0 && value <= 1.5;
    }

    return value >= 0 && value <= 1.5;
  });

  const scaleX = normalized ? snapshot.width : snapshot.width / MODEL_WIDTH;
  const scaleY = normalized ? snapshot.height : snapshot.height / MODEL_HEIGHT;
  const step = lastDim % 3 === 0 ? 3 : 2;
  const keypoints: FaceKeypoint[] = [];

  for (let offset = 0; offset + 1 < raw.length; offset += step) {
    keypoints.push({
      x: clamp(raw[offset] * scaleX, 0, snapshot.width),
      y: clamp(raw[offset + 1] * scaleY, 0, snapshot.height),
      score: step === 3 ? raw[offset + 2] : undefined,
    });
  }

  return keypoints;
}

function getChannelMajorValue(
  values: Float32Array,
  anchors: number,
  channel: number,
  anchorIndex: number,
) {
  return values[channel * anchors + anchorIndex] ?? 0;
}

function decodeDFLDistance(
  values: Float32Array,
  anchors: number,
  side: number,
  anchorIndex: number,
) {
  const logits: number[] = [];

  for (let bin = 0; bin < REG_MAX; bin += 1) {
    logits.push(
      getChannelMajorValue(values, anchors, side * REG_MAX + bin, anchorIndex),
    );
  }

  const probabilities = softmax(logits);
  let distance = 0;

  for (let bin = 0; bin < probabilities.length; bin += 1) {
    distance += probabilities[bin] * bin;
  }

  return distance;
}

function decodeSplitHeadBox(
  values: Float32Array,
  anchors: number,
  anchorIndex: number,
  gridX: number,
  gridY: number,
  stride: number,
  snapshot: SnapshotInfo,
): FaceBox {
  const left = decodeDFLDistance(values, anchors, 0, anchorIndex);
  const top = decodeDFLDistance(values, anchors, 1, anchorIndex);
  const right = decodeDFLDistance(values, anchors, 2, anchorIndex);
  const bottom = decodeDFLDistance(values, anchors, 3, anchorIndex);

  const centerX = (gridX + 0.5) * stride;
  const centerY = (gridY + 0.5) * stride;

  // model-space box
  const x1 = centerX - left * stride;
  const y1 = centerY - top * stride;
  const x2 = centerX + right * stride;
  const y2 = centerY + bottom * stride;

  // scale once into original photo space
  return {
    x1: clamp((x1 / MODEL_WIDTH) * snapshot.width, 0, snapshot.width),
    y1: clamp((y1 / MODEL_HEIGHT) * snapshot.height, 0, snapshot.height),
    x2: clamp((x2 / MODEL_WIDTH) * snapshot.width, 0, snapshot.width),
    y2: clamp((y2 / MODEL_HEIGHT) * snapshot.height, 0, snapshot.height),
  };
}

function decodeSplitHeadKeypoints(
  values: Float32Array | null,
  shape: number[] | null,
  anchorIndex: number,
  gridX: number,
  gridY: number,
  stride: number,
  snapshot: SnapshotInfo,
) {
  if (!values || !shape) {
    return [];
  }

  const channelCount = getSecondLastDim(shape);
  const anchorCount = getLastDim(shape);

  if (!channelCount || !anchorCount || anchorIndex >= anchorCount) {
    return [];
  }

  const pointStride = channelCount % 5 === 0 ? 5 : 3;
  const pointCount = Math.floor(channelCount / pointStride);
  const keypoints: FaceKeypoint[] = [];

  for (let pointIndex = 0; pointIndex < pointCount; pointIndex += 1) {
    const baseChannel = pointIndex * pointStride;
    const rawX = getChannelMajorValue(
      values,
      channelCount,
      anchorCount,
      baseChannel,
      anchorIndex,
    );
    const rawY = getChannelMajorValue(
      values,
      channelCount,
      anchorCount,
      baseChannel + 1,
      anchorIndex,
    );
    const decodedX = (rawX * 2 + gridX - 0.5) * stride;
    const decodedY = (rawY * 2 + gridY - 0.5) * stride;

    keypoints.push({
      x: clamp(decodedX * (snapshot.width / MODEL_WIDTH), 0, snapshot.width),
      y: clamp(decodedY * (snapshot.height / MODEL_HEIGHT), 0, snapshot.height),
      score:
        pointStride >= 3
          ? sigmoid(
              getChannelMajorValue(
                values,
                anchorCount,
                baseChannel + 2,
                anchorIndex,
              ),
            )
          : undefined,
    });
  }

  return keypoints;
}

function l2Normalize(vector: Float32Array) {
  let sum = 0;

  for (let i = 0; i < vector.length; i += 1) {
    sum += vector[i] * vector[i];
  }

  const norm = Math.sqrt(sum);

  if (!Number.isFinite(norm) || norm <= 0) {
    return vector;
  }

  const normalized = new Float32Array(vector.length);

  for (let i = 0; i < vector.length; i += 1) {
    normalized[i] = vector[i] / norm;
  }

  return normalized;
}

function cosineSimilarity(a: Float32Array, b: Float32Array) {
  if (a.length !== b.length || a.length === 0) {
    return 0;
  }

  let dot = 0;

  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
  }

  return dot;
}

function normalizeBox(rawBox: Float32Array, snapshot: SnapshotInfo) {
  const [rawX1 = 0, rawY1 = 0, rawX2 = 0, rawY2 = 0] = rawBox;
  const normalized = Math.max(rawX1, rawY1, rawX2, rawY2) <= 1.5;
  const scaleX = normalized ? snapshot.width : snapshot.width / MODEL_WIDTH;
  const scaleY = normalized ? snapshot.height : snapshot.height / MODEL_HEIGHT;

  const x1 = clamp(rawX1 * scaleX, 0, snapshot.width);
  const y1 = clamp(rawY1 * scaleY, 0, snapshot.height);
  const x2 = clamp(rawX2 * scaleX, 0, snapshot.width);
  const y2 = clamp(rawY2 * scaleY, 0, snapshot.height);

  return {
    x1: Math.min(x1, x2),
    y1: Math.min(y1, y2),
    x2: Math.max(x1, x2),
    y2: Math.max(y1, y2),
  };
}

function normalizeCenterBox(
  centerX: number,
  centerY: number,
  width: number,
  height: number,
  snapshot: SnapshotInfo,
) {
  const normalized =
    Math.max(
      Math.abs(centerX),
      Math.abs(centerY),
      Math.abs(width),
      Math.abs(height),
    ) <= 1.5;
  const scaleX = normalized ? snapshot.width : snapshot.width / MODEL_WIDTH;
  const scaleY = normalized ? snapshot.height : snapshot.height / MODEL_HEIGHT;
  const x1 = clamp((centerX - width / 2) * scaleX, 0, snapshot.width);
  const y1 = clamp((centerY - height / 2) * scaleY, 0, snapshot.height);
  const x2 = clamp((centerX + width / 2) * scaleX, 0, snapshot.width);
  const y2 = clamp((centerY + height / 2) * scaleY, 0, snapshot.height);

  return {
    x1: Math.min(x1, x2),
    y1: Math.min(y1, y2),
    x2: Math.max(x1, x2),
    y2: Math.max(y1, y2),
  };
}

function decodeCombinedKeypoints(
  values: Float32Array | null,
  shape: number[] | null,
  anchorIndex: number,
  snapshot: SnapshotInfo,
) {
  if (!values || !shape) {
    return [];
  }

  const channelCount = getSecondLastDim(shape);
  const anchorCount = getLastDim(shape);

  if (channelCount < 20 || anchorIndex >= anchorCount) {
    return [];
  }

  const keypoints: FaceKeypoint[] = [];

  for (let offset = 5; offset + 2 < channelCount; offset += 3) {
    const x = getCombinedTensorValue(
      values,
      channelCount,
      anchorCount,
      offset,
      anchorIndex,
    );
    const y = getCombinedTensorValue(
      values,
      channelCount,
      anchorCount,
      offset + 1,
      anchorIndex,
    );
    const score = getCombinedTensorValue(
      values,
      channelCount,
      anchorCount,
      offset + 2,
      anchorIndex,
    );
    [];
    keypoints.push({
      x: clamp(x * (snapshot.width / MODEL_WIDTH), 0, snapshot.width),
      y: clamp(y * (snapshot.height / MODEL_HEIGHT), 0, snapshot.height),
      score: score >= 0 && score <= 1 ? score : sigmoid(score),
    });
  }

  return keypoints;
}

function pickTensorByShape(
  tensors: { values: Float32Array; shape: number[] }[],
  predicate: (shape: number[]) => boolean,
) {
  const matchIndex = tensors.findIndex((tensor) => predicate(tensor.shape));

  if (matchIndex < 0) {
    return null;
  }

  return tensors.splice(matchIndex, 1)[0];
}

function parseFaceOutputCandidates(outputs: TensorPtr[]) {
  const tensors = outputs.map((output) => ({
    values: toFloatArray(output),
    shape: output.sizes ?? [],
  }));
  const candidates: ParsedFaceOutputs[] = [];

  const dflBoxTensor = tensors.find(
    (tensor) => getSecondLastDim(tensor.shape) === REG_MAX * 4,
  );
  const scoreTensor = tensors.find(
    (tensor) =>
      getSecondLastDim(tensor.shape) === 1 && getLastDim(tensor.shape) === 8400,
  );
  const keypointTensor = tensors.find(
    (tensor) =>
      getSecondLastDim(tensor.shape) === 15 &&
      getLastDim(tensor.shape) === 8400,
  );
  const combinedTensor = tensors.find(
    (tensor) =>
      getSecondLastDim(tensor.shape) === 20 &&
      getLastDim(tensor.shape) === 8400,
  );

  if (dflBoxTensor && scoreTensor) {
    candidates.push({
      boxes: dflBoxTensor.values,
      boxesShape: dflBoxTensor.shape,
      scores: scoreTensor.values,
      scoresShape: scoreTensor.shape,
      boxEncoding: "dfl",
      fusedScores: combinedTensor
        ? (() => {
            const anchorCount = getLastDim(combinedTensor.shape);
            const fusedScoreValues = new Float32Array(anchorCount);

            for (
              let anchorIndex = 0;
              anchorIndex < anchorCount;
              anchorIndex += 1
            ) {
              fusedScoreValues[anchorIndex] = getCombinedTensorValue(
                combinedTensor.values,
                20,
                anchorCount,
                4,
                anchorIndex,
              );
            }

            return fusedScoreValues;
          })()
        : null,
      fusedScoresShape: combinedTensor
        ? [1, 1, getLastDim(combinedTensor.shape)]
        : null,
      embeddings: null,
      embeddingsShape: null,
      keypoints: keypointTensor?.values ?? null,
      keypointsShape: keypointTensor?.shape ?? null,
    });
  }

  if (combinedTensor) {
    const anchorCount = getLastDim(combinedTensor.shape);
    const boxValues = new Float32Array(anchorCount * 4);
    const scoreValues = new Float32Array(anchorCount);
    const keypointValues = new Float32Array(anchorCount * 15);

    for (let anchorIndex = 0; anchorIndex < anchorCount; anchorIndex += 1) {
      for (let channel = 0; channel < 4; channel += 1) {
        boxValues[channel * anchorCount + anchorIndex] = getCombinedTensorValue(
          combinedTensor.values,
          20,
          anchorCount,
          channel,
          anchorIndex,
        );
      }

      scoreValues[anchorIndex] =
        scoreTensor?.values[anchorIndex] ??
        getCombinedTensorValue(
          combinedTensor.values,
          20,
          anchorCount,
          4,
          anchorIndex,
        );

      for (let channel = 0; channel < 15; channel += 1) {
        keypointValues[channel * anchorCount + anchorIndex] =
          getCombinedTensorValue(
            combinedTensor.values,
            20,
            anchorCount,
            channel + 5,
            anchorIndex,
          );
      }
    }

    candidates.push({
      boxes: boxValues,
      boxesShape: [1, 4, anchorCount],
      scores: scoreValues,
      scoresShape: [1, 1, anchorCount],
      boxEncoding: "xywh",
      fusedScores: null,
      fusedScoresShape: null,
      embeddings: null,
      embeddingsShape: null,
      keypoints: keypointValues,
      keypointsShape: [1, 15, anchorCount],
    });
  }

  const boxTensor = pickTensorByShape(tensors, isLikelyBoxes);
  const fallbackKeypointTensor = pickTensorByShape(tensors, isLikelyKeypoints);
  const fallbackEmbeddingTensor = pickTensorByShape(
    tensors,
    isLikelyEmbeddings,
  );
  const fallbackScoreTensor = tensors.shift() ?? null;

  if (boxTensor && fallbackScoreTensor) {
    candidates.push({
      boxes: boxTensor.values,
      boxesShape: boxTensor.shape,
      scores: fallbackScoreTensor.values,
      scoresShape: fallbackScoreTensor.shape,
      boxEncoding: "xyxy",
      fusedScores: null,
      fusedScoresShape: null,
      embeddings: fallbackEmbeddingTensor?.values ?? null,
      embeddingsShape: fallbackEmbeddingTensor?.shape ?? null,
      keypoints: fallbackKeypointTensor?.values ?? null,
      keypointsShape: fallbackKeypointTensor?.shape ?? null,
    });
  }

  return candidates;
}

function parseFaceOutputs(outputs: TensorPtr[]): ParsedFaceOutputs | null {
  return parseFaceOutputCandidates(outputs)[0] ?? null;
}

function decodeStructuredFaceOutputs(
  parsed: ParsedFaceOutputs,
  snapshot: SnapshotInfo,
  referenceFace: ReferenceFace | null,
) {
  const detections: Detection[] = [];

  if (parsed.boxEncoding === "dfl") {
    const anchorCount = getLastDim(parsed.boxesShape);
    let anchorIndex = 0;
    let bestScore = 0;
    let bestRawScore = 0;
    let bestSplitScore = 0;
    let bestBox: FaceBox | null = null;

    for (const stride of STRIDES) {
      const gridWidth = MODEL_WIDTH / stride;
      const gridHeight = MODEL_HEIGHT / stride;

      for (let gridY = 0; gridY < gridHeight; gridY += 1) {
        for (let gridX = 0; gridX < gridWidth; gridX += 1) {
          if (anchorIndex >= anchorCount) {
            break;
          }

          const rawScore = inferScoreAtIndex(
            parsed.scores,
            parsed.scoresShape,
            anchorIndex,
          );
          const fusedRawScore =
            parsed.fusedScores != null
              ? inferScoreAtIndex(
                  parsed.fusedScores,
                  parsed.fusedScoresShape ?? [],
                  anchorIndex,
                )
              : rawScore;
          const splitScore =
            rawScore >= 0 && rawScore <= 1 ? rawScore : sigmoid(rawScore);
          const fusedScore =
            fusedRawScore >= 0 && fusedRawScore <= 1
              ? fusedRawScore
              : sigmoid(fusedRawScore);
          const score = Math.max(splitScore, fusedScore);

          bestRawScore = Math.max(bestRawScore, fusedRawScore);
          bestSplitScore = Math.max(bestSplitScore, splitScore);
          if (score > bestScore) {
            bestScore = score;
          }

          if (!Number.isFinite(score) || score < CONF_THRESHOLD) {
            anchorIndex += 1;
            continue;
          }

          const bbox = decodeSplitHeadBox(
            parsed.boxes,
            anchorCount,
            anchorIndex,
            gridX,
            gridY,
            stride,
            snapshot,
          );

          if (bbox.x2 <= bbox.x1 || bbox.y2 <= bbox.y1) {
            anchorIndex += 1;
            continue;
          }

          if (score >= bestScore) {
            bestBox = bbox;
          }

          const embeddingSlice = sliceEmbedding(
            parsed.embeddings,
            parsed.embeddingsShape,
            anchorIndex,
          );
          const normalizedEmbedding = embeddingSlice
            ? l2Normalize(embeddingSlice)
            : null;
          const similarity =
            referenceFace && normalizedEmbedding
              ? cosineSimilarity(referenceFace.embedding, normalizedEmbedding)
              : null;
          const isRecognized =
            similarity != null && similarity >= RECOGNITION_THRESHOLD;

          const keypoints = decodeSplitHeadKeypoints(
            parsed.keypoints,
            parsed.keypointsShape,
            anchorIndex,
            gridX,
            gridY,
            stride,
            snapshot,
          );
          const faceBox = deriveFaceBoxFromKeypoints(keypoints, bbox, snapshot);

          detections.push({
            label: "Face",
            score,
            bbox: faceBox,
            keypoints,
            embedding: normalizedEmbedding,
            similarity,
            recognizedName: isRecognized
              ? (referenceFace?.name ?? "Known Face")
              : null,
          });

          anchorIndex += 1;
        }
      }
    }

    console.log(
      "[face] dfl best score:",
      bestScore,
      "best fused raw:",
      bestRawScore,
      "best split:",
      bestSplitScore,
      "best box:",
      bestBox,
    );

    return finalizeDetections(detections, snapshot);
  }

  if (parsed.boxEncoding === "xywh") {
    const anchorCount = getLastDim(parsed.boxesShape);
    let bestScore = 0;
    let bestRawBox: {
      centerX: number;
      centerY: number;
      width: number;
      height: number;
    } | null = null;
    let bestBox: FaceBox | null = null;

    for (let anchorIndex = 0; anchorIndex < anchorCount; anchorIndex += 1) {
      const rawScore = inferScoreAtIndex(
        parsed.scores,
        parsed.scoresShape,
        anchorIndex,
      );
      const score =
        rawScore >= 0 && rawScore <= 1 ? rawScore : sigmoid(rawScore);
      if (!Number.isFinite(score) || score < 0.2) {
        continue;
      }

      const centerX = getChannelMajorValue(
        parsed.boxes,
        anchorCount,
        0,
        anchorIndex,
      );
      const centerY = getChannelMajorValue(
        parsed.boxes,
        anchorCount,
        1,
        anchorIndex,
      );
      const width = getChannelMajorValue(
        parsed.boxes,
        anchorCount,
        2,
        anchorIndex,
      );
      const height = getChannelMajorValue(
        parsed.boxes,
        anchorCount,
        3,
        anchorIndex,
      );

      if (width <= 0 || height <= 0) {
        continue;
      }

      if (score >= bestScore) {
        bestScore = score;
        bestRawBox = {
          centerX,
          centerY,
          width,
          height,
        };
      }

      const bbox = normalizeCenterBox(
        centerX,
        centerY,
        width,
        height,
        snapshot,
      );

      if (bbox.x2 <= bbox.x1 || bbox.y2 <= bbox.y1) {
        continue;
      }

      if (score >= bestScore) {
        bestBox = bbox;
      }

      const keypoints = decodeCombinedKeypoints(
        parsed.keypoints,
        parsed.keypointsShape,
        anchorIndex,
        snapshot,
      );
      const faceBox = deriveFaceBoxFromKeypoints(keypoints, bbox, snapshot);

      detections.push({
        label: "Face",
        score,
        bbox: faceBox,
        keypoints,
        embedding: null,
        similarity: null,
        recognizedName: null,
      });
    }

    console.log(
      "[face] xywh best score:",
      bestScore,
      "best raw box:",
      bestRawBox,
      "best box:",
      bestBox,
    );

    return finalizeDetections(detections, snapshot);
  }

  const detectionCount = Math.min(
    getDetectionCount(parsed.boxesShape),
    getDetectionCount(parsed.scoresShape),
  );

  for (let index = 0; index < detectionCount; index += 1) {
    const score = inferScoreAtIndex(parsed.scores, parsed.scoresShape, index);

    if (!Number.isFinite(score) || score < CONF_THRESHOLD) {
      continue;
    }

    const rawBox = parsed.boxes.slice(index * 4, index * 4 + 4);

    if (rawBox.length < 4) {
      continue;
    }

    const bbox = normalizeBox(rawBox, snapshot);

    if (bbox.x2 <= bbox.x1 || bbox.y2 <= bbox.y1) {
      continue;
    }

    const embeddingSlice = sliceEmbedding(
      parsed.embeddings,
      parsed.embeddingsShape,
      index,
    );
    const normalizedEmbedding = embeddingSlice
      ? l2Normalize(embeddingSlice)
      : null;
    const similarity =
      referenceFace && normalizedEmbedding
        ? cosineSimilarity(referenceFace.embedding, normalizedEmbedding)
        : null;
    const isRecognized =
      similarity != null && similarity >= RECOGNITION_THRESHOLD;

    detections.push({
      label: "Face",
      score,
      bbox,
      keypoints: sliceKeypoints(
        parsed.keypoints,
        parsed.keypointsShape,
        index,
        snapshot,
      ),
      embedding: normalizedEmbedding,
      similarity,
      recognizedName: isRecognized
        ? (referenceFace?.name ?? "Known Face")
        : null,
    });
  }

  return finalizeDetections(detections, snapshot);
}

function describeOutputs(outputs: TensorPtr[]) {
  return outputs
    .map(
      (output, index) =>
        `#${String(index)} shape=${JSON.stringify(output.sizes ?? [])} type=${String(
          output.scalarType,
        )}`,
    )
    .join(" | ");
}

function getBestParsedScore(parsed: ParsedFaceOutputs) {
  const anchorCount =
    getLastDim(parsed.scoresShape) ||
    getLastDim(parsed.boxesShape) ||
    getDetectionCount(parsed.scoresShape);
  let bestScore = 0;

  for (let anchorIndex = 0; anchorIndex < anchorCount; anchorIndex += 1) {
    const rawScore = inferScoreAtIndex(
      parsed.fusedScores ?? parsed.scores,
      parsed.fusedScoresShape ?? parsed.scoresShape,
      anchorIndex,
    );
    const score = rawScore >= 0 && rawScore <= 1 ? rawScore : sigmoid(rawScore);
    bestScore = Math.max(bestScore, score);
  }

  return bestScore;
}

function getVariantQuality(faces: Detection[], snapshot: SnapshotInfo) {
  if (faces.length === 0) {
    return -1;
  }

  let maxAreaRatio = 0;
  let avgScore = 0;

  for (const face of faces) {
    const areaRatio =
      getBoxArea(face.bbox) / Math.max(1, snapshot.width * snapshot.height);
    maxAreaRatio = Math.max(maxAreaRatio, areaRatio);
    avgScore += face.score;
  }

  avgScore /= faces.length;

  const countPenalty = Math.max(0, faces.length - 3) * 0.05;

  return maxAreaRatio * 10 + avgScore - countPenalty;
}

export default function FaceDetectionScreen() {
  const cameraRef = useRef<React.ElementRef<typeof Camera> | null>(null);
  const runningRef = useRef(false);

  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice("front");
  const [detections, setDetections] = useState<TrackedDetection[]>([]);
  const [debug, setDebug] = useState("Loading model...");
  const [snapshotInfo, setSnapshotInfo] = useState<SnapshotInfo | null>(null);
  const [previewSize, setPreviewSize] = useState({ width: 0, height: 0 });
  const [referenceFace, setReferenceFace] = useState<ReferenceFace | null>(
    null,
  );

  const { forward, isReady, error } = useExecutorchModule({
    modelSource: YOLO_FACE_MODEL,
  });

  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission, requestPermission]);

  useEffect(() => {
    if (isReady) {
      setDebug("Model ready");
    }
  }, [isReady]);

  useEffect(() => {
    if (error) {
      setDebug(String(error));
    }
  }, [error]);

  const photoOutput = usePhotoOutput({
    targetResolution: {
      width: MODEL_WIDTH,
      height: MODEL_HEIGHT,
    },
  });

  const detectFaces = useCallback(async () => {
    if (!cameraRef.current || !device || !isReady || runningRef.current) {
      return;
    }

    let image: Image | null = null;
    let resized: Image | null = null;

    try {
      runningRef.current = true;
      setDebug("Capturing...");

      const { filePath } = await photoOutput.capturePhotoToFile(
        {
          enableAutoStabilization: true,
        },
        {
          onShutter: () => console.log("[face] photo captured"),
        },
      );

      await CameraRoll.save(filePath, { type: "photo" });

      image = await loadImage({ filePath });

      const snapshot = {
        width: image.width,
        height: image.height,
      };

      setSnapshotInfo(snapshot);
      setDebug("Preprocessing...");

      resized = await image.resizeAsync(MODEL_WIDTH, MODEL_HEIGHT);

      const raw = resized.toRawPixelData();
      const variants: TensorVariant[] = [
        { id: "rgb-0to1", channelOrder: "rgb", normalize: true },
        { id: "bgr-0to1", channelOrder: "bgr", normalize: true },
        { id: "rgb-0to255", channelOrder: "rgb", normalize: false },
        { id: "bgr-0to255", channelOrder: "bgr", normalize: false },
      ];

      let bestFaces: Detection[] = [];
      let bestVariantId = "";
      let bestDecoderId = "";
      let bestVariantScore = 0;
      let bestVariantQuality = -1;
      let bestOutputsDescription = "";

      for (const variant of variants) {
        setDebug(`Running inference (${variant.id})...`);

        const tensor = imageToFloatChw(raw.buffer, raw.pixelFormat, variant);
        const outputs = await forward([
          {
            dataPtr: tensor,
            sizes: [1, 3, MODEL_HEIGHT, MODEL_WIDTH],
            scalarType: ScalarType.FLOAT,
          },
        ]);

        if (!outputs?.length) {
          continue;
        }

        const outputsDescription = describeOutputs(outputs);
        console.log(`[face] outputs (${variant.id}):`, outputsDescription);

        const candidates = parseFaceOutputCandidates(outputs);

        if (!candidates.length) {
          continue;
        }

        for (const parsed of candidates) {
          const faces = decodeStructuredFaceOutputs(
            parsed,
            snapshot,
            referenceFace,
          );
          const parsedBestScore = getBestParsedScore(parsed);
          const variantQuality = getVariantQuality(faces, snapshot);

          console.log(
            `[face] variant ${variant.id}/${parsed.boxEncoding} => faces=${faces.length} best=${parsedBestScore} quality=${variantQuality}`,
          );

          if (
            variantQuality > bestVariantQuality ||
            (variantQuality === bestVariantQuality &&
              parsedBestScore > bestVariantScore)
          ) {
            bestFaces = faces;
            bestVariantId = variant.id;
            bestDecoderId = parsed.boxEncoding;
            bestVariantScore = parsedBestScore;
            bestVariantQuality = variantQuality;
            bestOutputsDescription = outputsDescription;
          }
        }
      }

      if (!bestVariantId) {
        throw new Error("Model returned no usable outputs");
      }

      setDetections(
        bestFaces.map((detection, index) => ({
          id: `${String(index)}-${Date.now()}`,
          detection,
        })),
      );

      const recognizedCount = bestFaces.filter(
        (face) => face.recognizedName != null,
      ).length;

      setDebug(
        `${bestFaces.length} face(s) | recognized: ${recognizedCount} | variant: ${bestVariantId}/${bestDecoderId} | best: ${bestVariantScore.toFixed(4)} | quality: ${bestVariantQuality.toFixed(3)} | outputs: ${bestOutputsDescription}`,
      );
    } catch (caughtError) {
      console.error(caughtError);
      setDebug(String(caughtError));
    } finally {
      resized?.dispose?.();
      image?.dispose?.();
      runningRef.current = false;
    }
  }, [device, forward, isReady, photoOutput, referenceFace]);

  const registerReferenceFace = useCallback(() => {
    const bestFace = [...detections]
      .map((item) => item.detection)
      .filter((item) => item.embedding != null)
      .sort((a, b) => b.score - a.score)[0];

    if (!bestFace?.embedding) {
      setDebug(
        "yolov8n-face detects faces and landmarks, but it does not provide identity embeddings for recognition",
      );
      return;
    }

    setReferenceFace({
      name: "Registered Face",
      embedding: bestFace.embedding,
    });
    setDebug("Reference face saved");
  }, [detections]);

  const boxes = useMemo(() => {
    if (!snapshotInfo || !previewSize.width || !previewSize.height) {
      return [];
    }

    const scale = Math.max(
      previewSize.width / snapshotInfo.width,
      previewSize.height / snapshotInfo.height,
    );
    const displayedWidth = snapshotInfo.width * scale;
    const displayedHeight = snapshotInfo.height * scale;
    const offsetX = (previewSize.width - displayedWidth) / 2;
    const offsetY = (previewSize.height - displayedHeight) / 2;

    return detections.map(({ id, detection }) => ({
      id,
      left: offsetX + detection.bbox.x1 * scale,
      top: offsetY + detection.bbox.y1 * scale,
      width: (detection.bbox.x2 - detection.bbox.x1) * scale,
      height: (detection.bbox.y2 - detection.bbox.y1) * scale,
      score: detection.score,
      name: detection.recognizedName ?? "Unknown",
      similarity: detection.similarity,
      keypoints: detection.keypoints.map((point, pointIndex) => ({
        id: `${id}-kp-${String(pointIndex)}`,
        left: offsetX + point.x * scale,
        top: offsetY + point.y * scale,
      })),
    }));
  }, [detections, previewSize, snapshotInfo]);

  if (!hasPermission) {
    return (
      <View style={styles.center}>
        <Text style={styles.text}>Camera permission required</Text>
      </View>
    );
  }

  if (!device) {
    return (
      <View style={styles.center}>
        <Text style={styles.text}>No camera found</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View
        style={styles.preview}
        onLayout={(event) => {
          const { width, height } = event.nativeEvent.layout;
          setPreviewSize({ width, height });
        }}
      >
        <Camera
          ref={cameraRef}
          style={StyleSheet.absoluteFill}
          device={device}
          isActive
          outputs={[photoOutput]}
        />

        <View pointerEvents="none" style={styles.overlay}>
          {boxes.map((box) => (
            <React.Fragment key={box.id}>
              <View
                style={[
                  styles.box,
                  {
                    left: box.left,
                    top: box.top,
                    width: box.width,
                    height: box.height,
                  },
                ]}
              >
                <View style={styles.badge}>
                  <Text style={styles.badgeText}>
                    {box.name} {(box.score * 100).toFixed(0)}%
                    {box.similarity != null
                      ? ` | sim ${(box.similarity * 100).toFixed(0)}%`
                      : ""}
                  </Text>
                </View>
              </View>

              {box.keypoints.map((point) => (
                <View
                  key={point.id}
                  style={[
                    styles.keypoint,
                    {
                      left: point.left - 3,
                      top: point.top - 3,
                    },
                  ]}
                />
              ))}
            </React.Fragment>
          ))}
        </View>

        <View style={styles.debug}>
          <Text style={styles.debugText}>{debug}</Text>
          <Text style={styles.debugText}>
            Reference: {referenceFace ? referenceFace.name : "None"}
          </Text>
        </View>

        <View style={styles.controls}>
          <TouchableOpacity
            style={[styles.button, !isReady && styles.buttonDisabled]}
            disabled={!isReady}
            onPress={() => {
              void detectFaces();
            }}
          >
            <Text style={styles.buttonText}>Detect Face</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.secondaryButton}
            onPress={registerReferenceFace}
          >
            <Text style={styles.buttonText}>Register Face</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.clearButton}
            onPress={() => {
              setReferenceFace(null);
              setDebug("Reference face cleared");
            }}
          >
            <Text style={styles.buttonText}>Clear Face</Text>
          </TouchableOpacity>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#000",
  },
  preview: {
    flex: 1,
  },
  center: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#000",
  },
  text: {
    color: "#fff",
  },
  overlay: {
    ...StyleSheet.absoluteFillObject,
  },
  box: {
    position: "absolute",
    borderWidth: 2,
    borderColor: "#00ff88",
    backgroundColor: "rgba(0,255,136,0.08)",
  },
  keypoint: {
    position: "absolute",
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: "#ffd60a",
  },
  badge: {
    backgroundColor: "#00ff88",
    alignSelf: "flex-start",
    paddingHorizontal: 6,
    paddingVertical: 2,
  },
  badgeText: {
    color: "#000",
    fontSize: 10,
    fontWeight: "700",
  },
  debug: {
    position: "absolute",
    top: 50,
    left: 10,
    right: 10,
    backgroundColor: "rgba(0,0,0,0.72)",
    padding: 10,
    borderRadius: 10,
    gap: 4,
  },
  debugText: {
    color: "#00ff88",
    fontSize: 12,
    fontFamily: "monospace",
  },
  controls: {
    position: "absolute",
    left: 16,
    right: 16,
    bottom: 40,
    gap: 12,
  },
  button: {
    alignSelf: "stretch",
    alignItems: "center",
    backgroundColor: "#2453ff",
    paddingHorizontal: 24,
    paddingVertical: 14,
    borderRadius: 12,
  },
  secondaryButton: {
    alignSelf: "stretch",
    alignItems: "center",
    backgroundColor: "#159957",
    paddingHorizontal: 24,
    paddingVertical: 14,
    borderRadius: 12,
  },
  clearButton: {
    alignSelf: "stretch",
    alignItems: "center",
    backgroundColor: "#6b7280",
    paddingHorizontal: 24,
    paddingVertical: 14,
    borderRadius: 12,
  },
  buttonDisabled: {
    opacity: 0.6,
  },
  buttonText: {
    color: "#fff",
    fontWeight: "700",
    fontSize: 16,
  },
});
