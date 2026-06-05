import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  Modal,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import {
  Detection,
  SSDLITE_320_MOBILENET_V3_LARGE,
  useObjectDetection,
} from "react-native-executorch";
import {
  Camera,
  CommonResolutions,
  Frame,
  useCameraDevice,
  useCameraPermission,
  useFrameOutput,
} from "react-native-vision-camera";
import { createSynchronizable, scheduleOnRN } from "react-native-worklets";

type CameraPosition = "back" | "front";
type ResolutionPreset = "hd" | "fhd";

type FrameSnapshot = {
  processedCount: number;
  width: number;
  height: number;
  orientation: string;
  isMirrored: boolean;
};
type TrackedDetection = {
  id: string;
  detection: Detection;
  staleFrames: number;
};

const cameraPositionSync = createSynchronizable<CameraPosition>("back");
const processedFramesSync = createSynchronizable(0);
const MAX_STALE_DETECTION_FRAMES = 12;
const DETECTION_SMOOTHING = 0.9;
const MIN_SCORE_TO_RENDER = 0.35;

function getBoxCenter(detection: Detection) {
  return {
    x: (detection.bbox.x1 + detection.bbox.x2) / 2,
    y: (detection.bbox.y1 + detection.bbox.y2) / 2,
  };
}

function getDetectionDistance(a: Detection, b: Detection) {
  const centerA = getBoxCenter(a);
  const centerB = getBoxCenter(b);
  return Math.hypot(centerA.x - centerB.x, centerA.y - centerB.y);
}

export default function ObjectDetectionScreen() {
  const { hasPermission, requestPermission } = useCameraPermission();
  const [cameraPosition, setCameraPosition] = useState<CameraPosition>("back");
  const [resolutionPreset, setResolutionPreset] =
    useState<ResolutionPreset>("hd");
  const [detectionThreshold, setDetectionThreshold] = useState(0.7);
  const [isSheetOpen, setIsSheetOpen] = useState(false);
  const [trackedDetections, setTrackedDetections] = useState<TrackedDetection[]>(
    [],
  );
  const [frameSnapshot, setFrameSnapshot] = useState<FrameSnapshot | null>(
    null,
  );
  const [previewSize, setPreviewSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    cameraPositionSync.setBlocking(cameraPosition);
  }, [cameraPosition]);

  useEffect(() => {
    processedFramesSync.setBlocking(0);
    setTrackedDetections([]);
    setFrameSnapshot(null);
  }, [cameraPosition, detectionThreshold, resolutionPreset]);

  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission, requestPermission]);

  const device = useCameraDevice(cameraPosition, {
    physicalDevices: ["ultra-wide-angle", "wide-angle", "telephoto"],
  });

  const { error, isReady, isGenerating, downloadProgress, runOnFrame } =
    useObjectDetection({ model: SSDLITE_320_MOBILENET_V3_LARGE });

  useEffect(() => {
    if (error) {
      console.warn("ExecuTorch object detection error:", error);
    }
  }, [error]);

  const targetResolution = useMemo(
    () =>
      resolutionPreset === "fhd"
        ? CommonResolutions.FHD_16_9
        : CommonResolutions.HD_16_9,
    [resolutionPreset],
  );

  const canRunDetection = isReady && runOnFrame != null;

  const handleFrameSnapshot = useCallback((snapshot: FrameSnapshot) => {
    setFrameSnapshot(snapshot);
  }, []);

  const handleDetections = useCallback((results: Detection[]) => {
    setTrackedDetections((previous) => {
      if (results.length === 0) {
        return previous
          .map((item) => ({
            ...item,
            staleFrames: item.staleFrames + 1,
          }))
          .filter((item) => item.staleFrames <= MAX_STALE_DETECTION_FRAMES);
      }

      const usedPreviousIds = new Set<string>();
      const nextTracked = results.map((result, index) => {
        const sameLabel = previous.filter(
          (item) =>
            item.detection.label === result.label &&
            !usedPreviousIds.has(item.id),
        );
        const match = sameLabel.sort(
          (a, b) =>
            getDetectionDistance(a.detection, result) -
            getDetectionDistance(b.detection, result),
        )[0];

        if (!match) {
          return {
            id: `${String(result.label)}-${index}-${Date.now()}`,
            detection: result,
            staleFrames: 0,
          };
        }

        usedPreviousIds.add(match.id);
        const prevBox = match.detection.bbox;
        const nextBox = result.bbox;

        return {
          id: match.id,
          staleFrames: 0,
          detection: {
            ...result,
            score:
              match.detection.score * DETECTION_SMOOTHING +
              result.score * (1 - DETECTION_SMOOTHING),
            bbox: {
              x1:
                prevBox.x1 * DETECTION_SMOOTHING +
                nextBox.x1 * (1 - DETECTION_SMOOTHING),
              y1:
                prevBox.y1 * DETECTION_SMOOTHING +
                nextBox.y1 * (1 - DETECTION_SMOOTHING),
              x2:
                prevBox.x2 * DETECTION_SMOOTHING +
                nextBox.x2 * (1 - DETECTION_SMOOTHING),
              y2:
                prevBox.y2 * DETECTION_SMOOTHING +
                nextBox.y2 * (1 - DETECTION_SMOOTHING),
            },
          },
        };
      });

      const staleCarryOver = previous
        .filter((item) => !usedPreviousIds.has(item.id))
        .map((item) => ({
          ...item,
          staleFrames: item.staleFrames + 1,
        }))
        .filter((item) => item.staleFrames <= MAX_STALE_DETECTION_FRAMES);

      return [...nextTracked, ...staleCarryOver];
    });
  }, []);

  const frameOutput = useFrameOutput({
    targetResolution,
    pixelFormat: "rgb",
    dropFramesWhileBusy: true,
    enablePreviewSizedOutputBuffers: true,
    allowDeferredStart: true,
    onFrame: useCallback(
      (frame: Frame) => {
        "worklet";

        const nextProcessedCount = processedFramesSync.getDirty() + 1;
        processedFramesSync.setBlocking(nextProcessedCount);

        const snapshot: FrameSnapshot = {
          processedCount: nextProcessedCount,
          width: frame.width,
          height: frame.height,
          orientation: frame.orientation,
          isMirrored: frame.isMirrored,
        };

        const shouldReportFrame =
          nextProcessedCount === 1 || nextProcessedCount % 12 === 0;

        if (!canRunDetection) {
          if (shouldReportFrame) {
            scheduleOnRN(handleFrameSnapshot, snapshot);
          }
          frame.dispose();
          return;
        }

        try {
          const isFrontCamera = cameraPositionSync.getDirty() === "front";
          const results =
            runOnFrame(frame, isFrontCamera, {
              detectionThreshold,
            }) ?? [];

          if (shouldReportFrame) {
            scheduleOnRN(handleFrameSnapshot, snapshot);
          }
          scheduleOnRN(handleDetections, results.slice(0, 10));
        } finally {
          frame.dispose();
        }
      },
      [
        canRunDetection,
        detectionThreshold,
        handleDetections,
        handleFrameSnapshot,
        runOnFrame,
      ],
    ),
  });

  const detectionBoxes = useMemo(() => {
    if (
      !frameSnapshot ||
      !previewSize.width ||
      !previewSize.height ||
      trackedDetections.length === 0
    ) {
      return [];
    }

    const orientation = frameSnapshot.orientation;
    const isQuarterTurn = orientation === "left" || orientation === "right";
    const sourceWidth = isQuarterTurn
      ? frameSnapshot.height
      : frameSnapshot.width;
    const sourceHeight = isQuarterTurn
      ? frameSnapshot.width
      : frameSnapshot.height;
    const previewWidth = previewSize.width;
    const previewHeight = previewSize.height;
    const scale = Math.max(
      previewWidth / sourceWidth,
      previewHeight / sourceHeight,
    );
    const scaledWidth = sourceWidth * scale;
    const scaledHeight = sourceHeight * scale;
    const offsetX = (previewWidth - scaledWidth) / 2;
    const offsetY = (previewHeight - scaledHeight) / 2;

    return trackedDetections
      .filter((item) => item.detection.score >= MIN_SCORE_TO_RENDER)
      .map((item) => {
        const detection = item.detection;
        const x1 = detection.bbox.x1 * scale + offsetX;
        const y1 = detection.bbox.y1 * scale + offsetY;
        const x2 = detection.bbox.x2 * scale + offsetX;
        const y2 = detection.bbox.y2 * scale + offsetY;
        const width = Math.max(0, x2 - x1);
        const height = Math.max(0, y2 - y1);
        const left = frameSnapshot.isMirrored ? previewWidth - x2 : x1;
        const top = y1;

        if (width < 8 || height < 8) return null;

        return {
          id: item.id,
          label: String(detection.label),
          score: detection.score,
          left: Math.max(0, Math.min(left, previewWidth - width)),
          top: Math.max(0, Math.min(top, previewHeight - height)),
          width: Math.min(width, previewWidth),
          height: Math.min(height, previewHeight),
        };
      })
      .filter((box): box is NonNullable<typeof box> => box != null);
  }, [frameSnapshot, previewSize, trackedDetections]);

  const detections = useMemo(
    () =>
      trackedDetections
        .map((item) => item.detection)
        .filter((item) => item.score >= MIN_SCORE_TO_RENDER),
    [trackedDetections],
  );

  if (!hasPermission) {
    return (
      <CenteredMessage
        title="Camera permission needed"
        message="This screen needs camera access to run live object detection."
        actionLabel="Grant Permission"
        onPress={requestPermission}
      />
    );
  }

  if (!device) {
    return (
      <CenteredMessage
        title="No camera device available"
        message="Try running this on a physical device or a simulator with camera support."
      />
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
          style={StyleSheet.absoluteFill}
          device={device}
          isActive={true}
          outputs={[frameOutput]}
          orientationSource="device"
        />

        <View style={styles.topRow}>
          <InfoPill
            label={cameraPosition === "back" ? "Back Camera" : "Front Camera"}
          />
          <InfoPill label="RGB Detection" />
          <InfoPill
            label={
              !isReady
                ? `Loading ${Math.round(downloadProgress * 100)}%`
                : isGenerating
                  ? "Detecting"
                  : "Ready"
            }
            accent={!isReady ? "#ffb020" : "#28c76f"}
          />
        </View>

        <View pointerEvents="none" style={styles.boxLayer}>
          {detectionBoxes.map((box) => (
            <View
              key={box.id}
              style={[
                styles.detectionBox,
                {
                  left: box.left,
                  top: box.top,
                  width: box.width,
                  height: box.height,
                },
              ]}
            >
              <View style={styles.boxBadge}>
                <Text style={styles.boxBadgeText}>
                  {box.label} {(box.score * 100).toFixed(0)}%
                </Text>
              </View>
            </View>
          ))}
        </View>

        <View style={styles.sidePanel}>
          <View style={styles.sideCard}>
            <Text style={styles.sideTitle}>
              {detections.length
                ? `${detections.length} objects`
                : "Object Detection"}
            </Text>
            <Text style={styles.sideMeta}>
              {frameSnapshot
                ? `${frameSnapshot.width} x ${frameSnapshot.height}`
                : "Waiting for frames"}
            </Text>
            <Text style={styles.sideMeta}>
              {frameSnapshot
                ? `${frameSnapshot.processedCount} processed`
                : "Initializing"}
            </Text>
          </View>
        </View>

        <View style={styles.bottomBar}>
          <TouchableOpacity
            style={styles.controlsButton}
            onPress={() => setIsSheetOpen(true)}
          >
            <Text style={styles.controlsButtonText}>Controls</Text>
          </TouchableOpacity>

          <View style={styles.detectedCount}>
            <Text style={styles.detectedCountText}>
              {detections.length} detected
            </Text>
          </View>
        </View>

        <View style={styles.ribbon}>
          <Text style={styles.ribbonTitle}>Detected Objects</Text>
          <ScrollView
            horizontal
            showsHorizontalScrollIndicator={false}
            contentContainerStyle={styles.ribbonContent}
          >
            {detections.length ? (
              detections.map((detection, index) => (
                <View
                  key={`${detection.label}-${index}`}
                  style={styles.ribbonChip}
                >
                  <Text style={styles.ribbonChipLabel} numberOfLines={1}>
                    {String(detection.label)}
                  </Text>
                  <Text style={styles.ribbonChipScore}>
                    {(detection.score * 100).toFixed(0)}%
                  </Text>
                </View>
              ))
            ) : (
              <View style={styles.ribbonEmpty}>
                <Text style={styles.ribbonEmptyText}>
                  {canRunDetection
                    ? "Point the camera at objects to see detections."
                    : "Model is loading. Detection starts automatically when ready."}
                </Text>
              </View>
            )}
          </ScrollView>
        </View>
      </View>

      <Modal
        visible={isSheetOpen}
        transparent
        animationType="slide"
        onRequestClose={() => setIsSheetOpen(false)}
      >
        <View style={styles.modalRoot}>
          <TouchableOpacity
            style={styles.backdrop}
            activeOpacity={1}
            onPress={() => setIsSheetOpen(false)}
          />
          <View style={styles.sheet}>
            <View style={styles.sheetHandle} />
            <View style={styles.sheetHeader}>
              <Text style={styles.sheetTitle}>Object Detection Controls</Text>
              <TouchableOpacity onPress={() => setIsSheetOpen(false)}>
                <Text style={styles.sheetClose}>Close</Text>
              </TouchableOpacity>
            </View>

            <ScrollView
              contentContainerStyle={styles.sheetContent}
              showsVerticalScrollIndicator={false}
            >
              <Section title="Camera">
                <Row>
                  <Chip
                    label="Back"
                    active={cameraPosition === "back"}
                    onPress={() => setCameraPosition("back")}
                  />
                  <Chip
                    label="Front"
                    active={cameraPosition === "front"}
                    onPress={() => setCameraPosition("front")}
                  />
                </Row>
              </Section>

              <Section title="Detection Settings">
                <Row>
                  <Chip
                    label="HD 16:9"
                    active={resolutionPreset === "hd"}
                    onPress={() => setResolutionPreset("hd")}
                  />
                  <Chip
                    label="FHD 16:9"
                    active={resolutionPreset === "fhd"}
                    onPress={() => setResolutionPreset("fhd")}
                  />
                </Row>
                <Row>
                  {[0.5, 0.7, 0.85].map((threshold) => (
                    <Chip
                      key={threshold}
                      label={`Confidence ${Math.round(threshold * 100)}%`}
                      active={detectionThreshold === threshold}
                      onPress={() => setDetectionThreshold(threshold)}
                    />
                  ))}
                </Row>
              </Section>

              <Section title="Detection Results">
                <Text style={styles.sheetStatus}>
                  {error
                    ? "Model error. Check the console for details."
                    : !isReady
                      ? `Loading model ${Math.round(downloadProgress * 100)}%`
                      : isGenerating
                        ? "Running inference on incoming frames."
                        : detections.length
                          ? `${detections.length} detections on the latest processed frame.`
                          : "Detection is ready. Point the camera at a scene."}
                </Text>
                {detections.length > 0 ? (
                  <View style={styles.resultList}>
                    {detections.map((detection, index) => (
                      <View
                        key={`${detection.label}-${index}-sheet`}
                        style={styles.resultRow}
                      >
                        <Text style={styles.resultLabel} numberOfLines={1}>
                          {String(detection.label)}
                        </Text>
                        <Text style={styles.resultScore}>
                          {(detection.score * 100).toFixed(1)}%
                        </Text>
                      </View>
                    ))}
                  </View>
                ) : null}
              </Section>
            </ScrollView>
          </View>
        </View>
      </Modal>
    </View>
  );
}

function CenteredMessage({
  title,
  message,
  actionLabel,
  onPress,
}: {
  title: string;
  message: string;
  actionLabel?: string;
  onPress?: () => void;
}) {
  return (
    <View style={styles.messageContainer}>
      <Text style={styles.messageTitle}>{title}</Text>
      <Text style={styles.messageText}>{message}</Text>
      {actionLabel && onPress ? (
        <TouchableOpacity style={styles.primaryButton} onPress={onPress}>
          <Text style={styles.primaryButtonText}>{actionLabel}</Text>
        </TouchableOpacity>
      ) : null}
    </View>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <View style={styles.section}>
      <Text style={styles.sectionTitle}>{title}</Text>
      {children}
    </View>
  );
}

function Row({ children }: { children: React.ReactNode }) {
  return <View style={styles.row}>{children}</View>;
}

function Chip({
  label,
  active,
  onPress,
}: {
  label: string;
  active: boolean;
  onPress: () => void;
}) {
  return (
    <TouchableOpacity
      style={[styles.chip, active && styles.chipActive]}
      onPress={onPress}
    >
      <Text style={[styles.chipText, active && styles.chipTextActive]}>
        {label}
      </Text>
    </TouchableOpacity>
  );
}

function InfoPill({
  label,
  accent = "#0a84ff",
}: {
  label: string;
  accent?: string;
}) {
  return (
    <View style={[styles.infoPill, { borderColor: accent }]}>
      <Text style={styles.infoPillText}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#000" },
  preview: { flex: 1, backgroundColor: "#000" },
  topRow: {
    position: "absolute",
    top: 14,
    left: 14,
    right: 14,
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
  },
  infoPill: {
    borderWidth: 1,
    borderRadius: 999,
    backgroundColor: "rgba(6, 18, 31, 0.58)",
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  infoPillText: {
    color: "#f4f7fb",
    fontSize: 12,
    fontWeight: "600",
  },
  boxLayer: {
    ...StyleSheet.absoluteFillObject,
  },
  detectionBox: {
    position: "absolute",
    borderWidth: 2,
    borderColor: "#4cff88",
    borderRadius: 12,
    backgroundColor: "rgba(76, 255, 136, 0.08)",
  },
  boxBadge: {
    position: "absolute",
    top: -2,
    left: -2,
    maxWidth: 160,
    backgroundColor: "#4cff88",
    borderTopLeftRadius: 10,
    borderBottomRightRadius: 10,
    paddingHorizontal: 8,
    paddingVertical: 5,
  },
  boxBadgeText: {
    color: "#06210f",
    fontSize: 11,
    fontWeight: "800",
  },
  sidePanel: {
    position: "absolute",
    top: 58,
    right: 14,
  },
  sideCard: {
    minWidth: 150,
    alignItems: "flex-end",
    gap: 2,
    borderRadius: 16,
    backgroundColor: "rgba(7, 20, 35, 0.58)",
    paddingHorizontal: 12,
    paddingVertical: 10,
  },
  sideTitle: {
    color: "#f7fbff",
    fontSize: 14,
    fontWeight: "800",
  },
  sideMeta: {
    color: "#a9bdd2",
    fontSize: 11,
    fontWeight: "600",
  },
  bottomBar: {
    position: "absolute",
    left: 14,
    right: 14,
    bottom: 118,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  controlsButton: {
    borderWidth: 1,
    borderColor: "#2f577e",
    borderRadius: 999,
    backgroundColor: "rgba(7, 20, 35, 0.82)",
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  controlsButtonText: {
    color: "#f5f8fc",
    fontSize: 13,
    fontWeight: "700",
  },
  detectedCount: {
    borderRadius: 999,
    backgroundColor: "rgba(7, 20, 35, 0.82)",
    paddingHorizontal: 14,
    paddingVertical: 10,
  },
  detectedCountText: {
    color: "#d7e2ef",
    fontSize: 12,
    fontWeight: "700",
  },
  ribbon: {
    position: "absolute",
    left: 12,
    right: 12,
    bottom: 18,
    gap: 8,
    borderRadius: 22,
    backgroundColor: "rgba(6, 18, 31, 0.84)",
    paddingVertical: 10,
    paddingLeft: 12,
  },
  ribbonTitle: {
    color: "#f4f8fc",
    fontSize: 12,
    fontWeight: "800",
    letterSpacing: 0.5,
  },
  ribbonContent: {
    gap: 8,
    paddingRight: 12,
  },
  ribbonChip: {
    minWidth: 112,
    maxWidth: 160,
    gap: 3,
    borderWidth: 1,
    borderColor: "#25496f",
    borderRadius: 16,
    backgroundColor: "#13243b",
    paddingHorizontal: 12,
    paddingVertical: 10,
  },
  ribbonChipLabel: {
    color: "#f7fbff",
    fontSize: 13,
    fontWeight: "700",
  },
  ribbonChipScore: {
    color: "#69d2ff",
    fontSize: 12,
    fontWeight: "700",
  },
  ribbonEmpty: {
    minWidth: 240,
    borderRadius: 16,
    backgroundColor: "#13243b",
    paddingHorizontal: 12,
    paddingVertical: 12,
  },
  ribbonEmptyText: {
    color: "#c3d1df",
    fontSize: 12,
    lineHeight: 17,
  },
  modalRoot: { flex: 1, justifyContent: "flex-end" },
  backdrop: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "rgba(0, 0, 0, 0.35)",
  },
  sheet: {
    position: "absolute",
    left: 0,
    right: 0,
    bottom: 0,
    maxHeight: "60%",
    borderTopLeftRadius: 22,
    borderTopRightRadius: 22,
    backgroundColor: "rgba(7, 15, 27, 0.97)",
  },
  sheetHandle: {
    alignSelf: "center",
    width: 44,
    height: 5,
    marginTop: 10,
    marginBottom: 10,
    borderRadius: 999,
    backgroundColor: "#5a6d84",
  },
  sheetHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 16,
    paddingBottom: 8,
  },
  sheetTitle: {
    color: "#f5f8fc",
    fontSize: 16,
    fontWeight: "800",
  },
  sheetClose: {
    color: "#69d2ff",
    fontSize: 14,
    fontWeight: "700",
  },
  sheetContent: {
    gap: 14,
    paddingTop: 4,
    paddingBottom: 20,
    paddingHorizontal: 14,
  },
  section: {
    gap: 12,
    borderRadius: 18,
    backgroundColor: "#101b2d",
    padding: 14,
  },
  sectionTitle: {
    color: "#f5f8fc",
    fontSize: 15,
    fontWeight: "700",
  },
  row: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
  },
  chip: {
    borderWidth: 1,
    borderColor: "#29415e",
    borderRadius: 999,
    backgroundColor: "#132238",
    paddingHorizontal: 14,
    paddingVertical: 10,
  },
  chipActive: {
    borderColor: "#4cc9f0",
    backgroundColor: "#11344b",
  },
  chipText: {
    color: "#c4d0df",
    fontSize: 12,
    fontWeight: "600",
  },
  chipTextActive: {
    color: "#f6fbff",
  },
  sheetStatus: {
    color: "#d7e2ef",
    fontSize: 13,
    lineHeight: 18,
  },
  resultList: { gap: 10 },
  resultRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 12,
    borderRadius: 16,
    backgroundColor: "#15253d",
    paddingHorizontal: 14,
    paddingVertical: 12,
  },
  resultLabel: {
    flex: 1,
    color: "#f4f7fb",
    fontSize: 14,
    fontWeight: "600",
  },
  resultScore: {
    color: "#63d2ff",
    fontSize: 14,
    fontWeight: "700",
  },
  messageContainer: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: 12,
    backgroundColor: "#09111f",
    paddingHorizontal: 28,
  },
  messageTitle: {
    color: "#f6fbff",
    fontSize: 22,
    fontWeight: "800",
    textAlign: "center",
  },
  messageText: {
    color: "#a8b6c7",
    fontSize: 15,
    lineHeight: 22,
    textAlign: "center",
  },
  primaryButton: {
    marginTop: 8,
    borderRadius: 999,
    backgroundColor: "#0a84ff",
    paddingHorizontal: 18,
    paddingVertical: 12,
  },
  primaryButtonText: {
    color: "#fff",
    fontSize: 14,
    fontWeight: "700",
  },
});
