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
  Camera,
  CommonResolutions,
  Frame,
  useCameraDevice,
  useCameraPermission,
  useFrameOutput,
} from "react-native-vision-camera";
import { createSynchronizable, scheduleOnRN } from "react-native-worklets";

type CameraPosition = "back" | "front";
type FrameOutputPixelFormat = "rgb" | "yuv" | "native";
type ResolutionPreset = "hd" | "fhd";

type FrameSnapshot = {
  processedCount: number;
  width: number;
  height: number;
  pixelFormat: string;
  orientation: string;
  isMirrored: boolean;
  isPlanar: boolean;
};

type RedBox = {
  x: number;
  y: number;
  width: number;
  height: number;
} | null;

const processedFramesSync = createSynchronizable<number>(0);
const previousFrameBuffer = createSynchronizable<Uint8Array | null>(null);
const motionValueSync = createSynchronizable<number>(0);

export default function FrameOutputLabScreen(): React.JSX.Element {
  const { hasPermission, requestPermission } = useCameraPermission();
  const [cameraPosition, setCameraPosition] = useState<CameraPosition>("back");
  const [pixelFormat, setPixelFormat] = useState<FrameOutputPixelFormat>("rgb");
  const [resolutionPreset, setResolutionPreset] =
    useState<ResolutionPreset>("hd");
  const [dropFramesWhileBusy, setDropFramesWhileBusy] = useState<boolean>(true);
  const [previewSizedBuffers, setPreviewSizedBuffers] = useState<boolean>(true);
  const [physicalRotation, setPhysicalRotation] = useState<boolean>(false);
  const [deferredStart, setDeferredStart] = useState<boolean>(true);
  const [isSheetOpen, setIsSheetOpen] = useState<boolean>(false);
  const [frameSnapshot, setFrameSnapshot] = useState<FrameSnapshot | null>(
    null,
  );
  const [droppedFrames, setDroppedFrames] = useState<number>(0);
  const [lastDropReason, setLastDropReason] = useState<string | null>(null);
  const [motionValue, setMotionValue] = useState<number>(0);
  const [redBox, setRedBox] = useState<RedBox>(null);

  useEffect(() => {
    processedFramesSync.setBlocking(0);
    setFrameSnapshot(null);
    setDroppedFrames(0);
    setLastDropReason(null);
  }, [
    cameraPosition,
    deferredStart,
    dropFramesWhileBusy,
    physicalRotation,
    pixelFormat,
    previewSizedBuffers,
    resolutionPreset,
  ]);

  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission, requestPermission]);

  const device = useCameraDevice(cameraPosition, {
    physicalDevices: ["ultra-wide-angle", "wide-angle", "telephoto"],
  });

  const targetResolution = useMemo(
    () =>
      resolutionPreset === "fhd"
        ? CommonResolutions.FHD_16_9
        : CommonResolutions.HD_16_9,
    [resolutionPreset],
  );

  const handleFrameSnapshot = useCallback((snapshot: FrameSnapshot): void => {
    setFrameSnapshot(snapshot);
  }, []);

  const frameOutput = useFrameOutput({
    targetResolution,
    pixelFormat,
    dropFramesWhileBusy,
    enablePreviewSizedOutputBuffers: previewSizedBuffers,
    enablePhysicalBufferRotation: physicalRotation,
    allowDeferredStart: deferredStart,
    onFrameDropped: useCallback((reason: string): void => {
      setDroppedFrames((value) => value + 1);
      setLastDropReason(reason);
    }, []),
    onFrame: useCallback(
      (frame: Frame): void => {
        "worklet";
        if (frame.isPlanar) {
          frame.dispose();
          return;
        }

        const buffer = frame.getPixelBuffer();
        const data = new Uint8Array(buffer);
        const prev = previousFrameBuffer.getDirty();

        let diff = 0;
        let samples = 0;

        // 🔴 RED DETECTION VARIABLES
        let minX = Number.MAX_VALUE;
        let minY = Number.MAX_VALUE;
        let maxX = 0;
        let maxY = 0;
        let redPixelCount = 0;
        const width = frame.width;

        const step = 32; // smaller = more accurate, bigger = faster
        const sampleCount = Math.ceil(data.length / (4 * step));
        const sampledValues = new Uint8Array(sampleCount);

        for (
          let i = 0, sampleIndex = 0;
          i < data.length;
          i += 4 * step, sampleIndex++
        ) {
          const r = data[i];
          const g = data[i + 1];
          const b = data[i + 2];

          // ✅ Grayscale (existing motion detection logic)
          const gray = 0.299 * r + 0.587 * g + 0.114 * b;
          if (prev) {
            diff += Math.abs(gray - prev[sampleIndex]);
          }
          sampledValues[sampleIndex] = Math.min(255, Math.round(gray));
          samples++;

          // 🔴 SIMPLE RED DETECTION (fast heuristic)
          if (r > 150 && g < 100 && b < 100 && r > g * 1.3 && r > b * 1.3) {
            redPixelCount++;
            const pixelIndex = i / 4;
            const x = pixelIndex % width;
            const y = Math.floor(pixelIndex / width);
            if (x < minX) minX = x;
            if (y < minY) minY = y;
            if (x > maxX) maxX = x;
            if (y > maxY) maxY = y;
          }
        }

        const avgDiff = diff / (samples || 1);
        previousFrameBuffer.setBlocking(sampledValues);
        motionValueSync.setBlocking(avgDiff);

        const nextProcessedCount = processedFramesSync.getDirty() + 1;
        processedFramesSync.setBlocking(nextProcessedCount);

        const shouldReportFrame =
          nextProcessedCount === 1 || nextProcessedCount % 12 === 0;

        if (shouldReportFrame) {
          scheduleOnRN(handleFrameSnapshot, {
            processedCount: nextProcessedCount,
            width: frame.width,
            height: frame.height,
            pixelFormat: frame.pixelFormat,
            orientation: frame.orientation,
            isMirrored: frame.isMirrored,
            isPlanar: frame.isPlanar,
          });
          scheduleOnRN(setMotionValue, avgDiff);

          // 🔴 SEND RED BOX
          if (redPixelCount > 50) {
            scheduleOnRN(setRedBox, {
              x: minX,
              y: minY,
              width: maxX - minX,
              height: maxY - minY,
            });
          } else {
            scheduleOnRN(setRedBox, null);
          }
        }

        frame.dispose();
      },
      [handleFrameSnapshot],
    ),
  });

  if (!hasPermission) {
    return (
      <CenteredMessage
        title="Camera permission needed"
        message="This screen needs camera access to inspect frame output."
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
      <View style={styles.preview}>
        <Camera
          style={StyleSheet.absoluteFill}
          device={device}
          isActive={true}
          outputs={[frameOutput]}
          orientationSource="device"
        />

        {redBox && (
          <View
            style={{
              position: "absolute",
              left: redBox.x,
              top: redBox.y,
              width: redBox.width,
              height: redBox.height,
              borderWidth: 2,
              borderColor: "red",
            }}
          />
        )}

        <View style={styles.topRow}>
          <InfoPill
            label={cameraPosition === "back" ? "Back Camera" : "Front Camera"}
          />
          <InfoPill label={`${pixelFormat.toUpperCase()} Output`} />
          <InfoPill
            label={
              dropFramesWhileBusy ? "Drop Busy Frames" : "Queue Busy Frames"
            }
            accent="#ffb020"
          />
        </View>

        <View style={styles.metricsColumn}>
          <MetricCard
            label="Frame"
            value={
              frameSnapshot
                ? `${frameSnapshot.width} x ${frameSnapshot.height}`
                : "Waiting"
            }
          />
          <MetricCard
            label="Motion"
            value={motionValue.toFixed(2)}
            hint={motionValue > 12 ? "Motion detected" : "Stable"}
          />
          <MetricCard
            label="Actual Format"
            value={frameSnapshot?.pixelFormat ?? "Waiting"}
          />
          <MetricCard
            label="Processed"
            value={String(frameSnapshot?.processedCount ?? 0)}
          />
          <MetricCard
            label="Dropped"
            value={String(droppedFrames)}
            hint={lastDropReason ?? "No drops yet"}
          />
        </View>

        <View style={styles.bottomBar}>
          <TouchableOpacity
            style={styles.controlsButton}
            onPress={() => setIsSheetOpen(true)}
          >
            <Text style={styles.controlsButtonText}>Controls</Text>
          </TouchableOpacity>
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
              <Text style={styles.sheetTitle}>Frame Output Controls</Text>
              <TouchableOpacity onPress={() => setIsSheetOpen(false)}>
                <Text style={styles.sheetClose}>Close</Text>
              </TouchableOpacity>
            </View>

            <ScrollView
              contentContainerStyle={styles.sheetContent}
              showsVerticalScrollIndicator={false}
            >
              <Section title="Frame Output Settings">
                <Row>
                  <Chip
                    label="RGB"
                    active={pixelFormat === "rgb"}
                    onPress={() => setPixelFormat("rgb")}
                  />
                  <Chip
                    label="YUV"
                    active={pixelFormat === "yuv"}
                    onPress={() => setPixelFormat("yuv")}
                  />
                  <Chip
                    label="Native"
                    active={pixelFormat === "native"}
                    onPress={() => setPixelFormat("native")}
                  />
                </Row>
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
              </Section>

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

              <Section title="Performance Controls">
                <Row>
                  <Chip
                    label={
                      previewSizedBuffers
                        ? "Preview Buffers On"
                        : "Preview Buffers Off"
                    }
                    active={previewSizedBuffers}
                    onPress={() => setPreviewSizedBuffers((value) => !value)}
                  />
                  <Chip
                    label={
                      dropFramesWhileBusy
                        ? "Drop Busy Frames On"
                        : "Drop Busy Frames Off"
                    }
                    active={dropFramesWhileBusy}
                    onPress={() => setDropFramesWhileBusy((value) => !value)}
                  />
                </Row>
                <Row>
                  <Chip
                    label={
                      deferredStart ? "Deferred Start On" : "Deferred Start Off"
                    }
                    active={deferredStart}
                    onPress={() => setDeferredStart((value) => !value)}
                  />
                  <Chip
                    label={
                      physicalRotation
                        ? "Physical Rotation On"
                        : "Physical Rotation Off"
                    }
                    active={physicalRotation}
                    onPress={() => setPhysicalRotation((value) => !value)}
                  />
                </Row>
              </Section>

              <Section title="Live Metadata">
                <InfoRow
                  label="Orientation"
                  value={frameSnapshot?.orientation ?? "Waiting"}
                />
                <InfoRow
                  label="Mirrored"
                  value={
                    frameSnapshot ? String(frameSnapshot.isMirrored) : "Waiting"
                  }
                />
                <InfoRow
                  label="Planar"
                  value={
                    frameSnapshot ? String(frameSnapshot.isPlanar) : "Waiting"
                  }
                />
                <InfoRow
                  label="Last Drop Reason"
                  value={lastDropReason ?? "None"}
                />
              </Section>
            </ScrollView>
          </View>
        </View>
      </Modal>
    </View>
  );
}

// ─── Sub-components ───────────────────────────────────────────────────────────

interface CenteredMessageProps {
  title: string;
  message: string;
  actionLabel?: string;
  onPress?: () => void;
}

function CenteredMessage({
  title,
  message,
  actionLabel,
  onPress,
}: CenteredMessageProps): React.JSX.Element {
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

interface SectionProps {
  title: string;
  children: React.ReactNode;
}

function Section({ title, children }: SectionProps): React.JSX.Element {
  return (
    <View style={styles.section}>
      <Text style={styles.sectionTitle}>{title}</Text>
      {children}
    </View>
  );
}

interface RowProps {
  children: React.ReactNode;
}

function Row({ children }: RowProps): React.JSX.Element {
  return <View style={styles.row}>{children}</View>;
}

interface ChipProps {
  label: string;
  active: boolean;
  onPress: () => void;
}

function Chip({ label, active, onPress }: ChipProps): React.JSX.Element {
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

interface InfoPillProps {
  label: string;
  accent?: string;
}

function InfoPill({
  label,
  accent = "#0a84ff",
}: InfoPillProps): React.JSX.Element {
  return (
    <View style={[styles.infoPill, { borderColor: accent }]}>
      <Text style={styles.infoPillText}>{label}</Text>
    </View>
  );
}

interface MetricCardProps {
  label: string;
  value: string;
  hint?: string;
}

function MetricCard({
  label,
  value,
  hint,
}: MetricCardProps): React.JSX.Element {
  return (
    <View style={styles.metricCard}>
      <Text style={styles.metricLabel}>{label}</Text>
      <Text style={styles.metricValue} numberOfLines={1}>
        {value}
      </Text>
      {hint ? <Text style={styles.metricHint}>{hint}</Text> : null}
    </View>
  );
}

interface InfoRowProps {
  label: string;
  value: string;
}

function InfoRow({ label, value }: InfoRowProps): React.JSX.Element {
  return (
    <View style={styles.infoRow}>
      <Text style={styles.infoLabel}>{label}</Text>
      <Text style={styles.infoValue}>{value}</Text>
    </View>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────

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
  metricsColumn: {
    position: "absolute",
    right: 14,
    top: 58,
    width: 172,
    gap: 10,
  },
  metricCard: {
    borderRadius: 16,
    backgroundColor: "rgba(7, 20, 35, 0.58)",
    padding: 10,
    justifyContent: "space-between",
  },
  metricLabel: {
    color: "#8fa2ba",
    fontSize: 11,
    textTransform: "uppercase",
    letterSpacing: 0.8,
  },
  metricValue: {
    color: "#f8fbff",
    fontSize: 14,
    fontWeight: "700",
  },
  metricHint: {
    color: "#7e90a6",
    fontSize: 11,
  },
  bottomBar: {
    position: "absolute",
    left: 14,
    right: 14,
    bottom: 24,
    flexDirection: "row",
    justifyContent: "flex-start",
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
  infoRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    gap: 16,
  },
  infoLabel: {
    color: "#8ea1b8",
    fontSize: 13,
  },
  infoValue: {
    flex: 1,
    textAlign: "right",
    color: "#f5f8fc",
    fontSize: 13,
    fontWeight: "600",
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
