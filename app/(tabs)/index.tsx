/**
 * ============================================================
 *  VisionCamera — Comprehensive Feature Demo
 *  react-native-vision-camera (Nitro / V4+ API)
 *
 *  ✅ Bug fix from original: videoOutput now uses useVideoOutput()
 *     not usePhotoOutput()
 *  ✅ Removed: useFrameOutput, react-native-vision-camera-worklets,
 *     react-native-worklets-core (no longer needed)
 *
 *  Covers:
 *   • useCameraPermission / useMicrophonePermission
 *   • useCameraDevice / useCameraDevices
 *   • usePhotoOutput / useVideoOutput
 *   • <Camera /> props: isActive, device, outputs, zoom,
 *       exposure, constraints (fps), orientation,
 *       enableNativeZoomGesture, enableNativeTapToFocusGesture,
 *       onConfigured, onError, onStarted, onStopped, resizeMode
 *   • CameraRef: focusTo()
 *   • photoOutput.capturePhotoToFile()
 *   • videoOutput.createRecorder() + recorder.startRecording()
 *     + recorder.stopRecording()
 *   • device capabilities: minZoom, maxZoom, supportedFPSRanges,
 *       supportsPhotoHDR, isVirtualDevice, physicalDevices
 *   • Reanimated SharedValue zoom
 * ============================================================
 */

import { useRouter } from "expo-router";
import React, { useCallback, useEffect, useRef, useState } from "react";
import {
    Alert,
    GestureResponderEvent,
    ScrollView,
    StyleSheet,
    Text,
    TouchableOpacity,
    View,
} from "react-native";
import {
    Camera,
    CameraDevice,
    CameraRef,
    CommonResolutions,
    useCameraDevice,
    useCameraDevices,
    useCameraPermission,
    useMicrophonePermission,
    usePhotoOutput,
    useVideoOutput,
} from "react-native-vision-camera";

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────
type Position = "back" | "front";
type ResizeMode = "cover" | "contain";
type FPS = 30 | 60 | 120;

// ─────────────────────────────────────────────
// Main Component
// ─────────────────────────────────────────────
export default function HomeScreen() {
  // ── 1. PERMISSIONS ──────────────────────────────────────────
  /**
   * useCameraPermission()
   *  • hasPermission: boolean  — whether camera permission is granted
   *  • requestPermission: () => Promise<boolean>  — prompts the user
   *
   * useMicrophonePermission()
   *  Same shape, but for microphone (needed for video with audio).
   */
  const router = useRouter();

  const { hasPermission: hasCamPerm, requestPermission: requestCamPerm } =
    useCameraPermission();

  const { hasPermission: hasMicPerm, requestPermission: requestMicPerm } =
    useMicrophonePermission();

  useEffect(() => {
    if (!hasCamPerm) requestCamPerm();
    if (!hasMicPerm) requestMicPerm();
  }, [hasCamPerm, hasMicPerm, requestCamPerm, requestMicPerm]);

  // ── 2. DEVICE SELECTION ─────────────────────────────────────
  /**
   * useCameraDevices()
   *  Returns ALL available CameraDevice objects on the system.
   *  Reactive: updates when an external camera is plugged in/out.
   */
  const allDevices: CameraDevice[] = useCameraDevices();

  /**
   * useCameraDevice(position, options?)
   *  Convenience hook: picks the best device for the given position.
   *
   *  position: 'back' | 'front' | 'external'
   *
   *  options.physicalDevices: string[]
   *    — Prefer a virtual device that combines the listed physical lenses.
   *      e.g. ['ultra-wide-angle', 'wide-angle', 'telephoto'] selects the
   *      triple-camera on supported iPhones.
   *
   *  The returned CameraDevice exposes:
   *    device.minZoom               — minimum zoom factor (e.g. 0.5)
   *    device.maxZoom               — maximum zoom factor (e.g. 15)
   *    device.neutralZoom           — "1x" zoom relative to the device
   *    device.zoomLensSwitchFactors — zoom levels where virtual device
   *                                   switches physical lenses
   *    device.supportedFPSRanges    — [{minFps, maxFps}, ...]
   *    device.supportsPhotoHDR      — boolean
   *    device.supportsVideoHDR      — boolean
   *    device.isVirtualDevice       — boolean (multi-lens)
   *    device.physicalDevices       — constituent physical devices
   *    device.getSupportedResolutions('photo' | 'video')
   */
  const [position, setPosition] = useState<Position>("back");

  const device = useCameraDevice(position, {
    physicalDevices: ["ultra-wide-angle", "wide-angle", "telephoto"],
  });

  // ── 3. OUTPUTS ───────────────────────────────────────────────
  /**
   * usePhotoOutput(options?)
   *  Creates a CameraPhotoOutput.
   *
   *  options.targetResolution — target capture resolution.
   *    Use CommonResolutions.UHD_16_9 (3840×2160), FHD_16_9 (1920×1080), etc.
   *
   *  Exposes:
   *    photoOutput.capturePhoto(settings, callbacks) → Photo (in-memory)
   *    photoOutput.capturePhotoToFile(settings, callbacks) → { filePath }
   *
   *  CapturePhotoSettings:
   *    flash: 'on' | 'off' | 'auto'
   *    enableAutoRedEyeReduction: boolean
   *    enableAutoStabilization: boolean
   *    quality: number (0–1)  [iOS only]
   *
   *  CapturePhotoCallbacks:
   *    onShutter — fires when the shutter fires
   */
  const photoOutput = usePhotoOutput({
    targetResolution: CommonResolutions.FHD_16_9, // 1920×1080
  });

  /**
   * useVideoOutput(options?)
   *  Creates a CameraVideoOutput.
   *
   *  ✅ Bug fix: original code used usePhotoOutput() here — now corrected.
   *
   *  options.targetResolution         — target recording resolution
   *  options.enableAudio              — record microphone audio (default: false)
   *  options.enablePersistentRecorder — keeps recording even when device
   *                                     flips front ↔ back (default: false)
   *  options.fileType                 — 'mov' (iOS default) | 'mp4'
   *
   *  Exposes:
   *    videoOutput.createRecorder(settings) → Recorder
   *      Recorder.startRecording(onFinished, onError)
   *      Recorder.stopRecording()
   *      Recorder.pauseRecording()
   *      Recorder.resumeRecording()
   */
  const videoOutput = useVideoOutput({
    targetResolution: CommonResolutions.FHD_16_9,
    enableAudio: hasMicPerm, // only if microphone permission is granted
    enablePersistentRecorder: false,
    fileType: "mp4",
  });

  // ── 4. ZOOM ───────────────────────────────────────────────
  /**
   * Zoom can be driven by a plain number via the `zoom` prop.
   * This avoids the worklets runtime dependency when no frame processors
   * are in use.
   *
   * Always clamp to [device.minZoom, device.maxZoom].
   */
  const [zoom, setZoom] = useState(1);
  const [zoomDisplay, setZoomDisplay] = useState(1);

  const updateZoom = useCallback(
    (value: number) => {
      if (!device) return;
      const clamped = Math.min(Math.max(value, device.minZoom), device.maxZoom);
      setZoom(clamped);
      setZoomDisplay(clamped);
    },
    [device],
  );

  // ── 5. EXPOSURE BIAS ────────────────────────────────────────
  /**
   * exposure prop: number
   *  Adjusts the exposure bias in EV stops.
   *  Range is typically -3 to +3 (device-dependent).
   *  Can also be a Reanimated SharedValue<number>.
   */
  const [exposure, setExposure] = useState(0); // 0 = neutral

  // ── 6. FPS CONSTRAINT ───────────────────────────────────────
  /**
   * constraints prop: Constraint[]
   *  Passes constraints to the camera session.
   *  { fps: number } requests a specific frame rate.
   *  VisionCamera negotiates the best available configuration.
   */
  const [targetFps, setTargetFps] = useState<FPS>(30);

  // ── 7. ORIENTATION ──────────────────────────────────────────
  /**
   * orientation prop: 'portrait' | 'portraitUpsideDown' |
   *                   'landscapeLeft' | 'landscapeRight'
   *  Controls the camera preview orientation.
   *  Default: follows device orientation automatically.
   */
  const [orientation, setOrientation] = useState<"portrait" | "landscapeLeft">(
    "portrait",
  );

  // ── 8. RESIZE MODE ──────────────────────────────────────────
  /**
   * resizeMode prop: 'cover' | 'contain'
   *  'cover'   — fills the view, may crop (like CSS background-size: cover)
   *  'contain' — letterboxed, shows the full frame
   */
  const [resizeMode, setResizeMode] = useState<ResizeMode>("cover");

  // ── 9. NATIVE GESTURES ──────────────────────────────────────
  /**
   * enableNativeZoomGesture: boolean
   *  Enables the built-in pinch-to-zoom gesture (ZoomGestureController).
   *  No external gesture library needed.
   *
   * enableNativeTapToFocusGesture: boolean
   *  Enables the built-in tap-to-focus gesture (TapToFocusGestureController).
   *  Automatically converts view coordinates to camera coordinates.
   */
  const [nativeZoom, setNativeZoom] = useState(false);
  const [nativeTapFocus, setNativeTapFocus] = useState(false);

  // ── 10. CameraRef — imperative focus ────────────────────────
  /**
   * useRef<CameraRef>()
   *  Attach to <Camera ref={cameraRef} /> for imperative control.
   *
   *  CameraRef exposes:
   *    cameraRef.current.focusTo({ x, y })
   *      — starts a Focus Metering Action at the given view-coordinate point.
   *        Converts view coords → camera coords internally.
   */
  const cameraRef = useRef<CameraRef>(null);

  const handleManualTapFocus = useCallback(
    async (e: GestureResponderEvent) => {
      if (nativeTapFocus) return; // let the native gesture handle it
      const { locationX, locationY } = e.nativeEvent;
      try {
        await cameraRef.current?.focusTo({ x: locationX, y: locationY });
        console.log(
          `Focused at (${locationX.toFixed(0)}, ${locationY.toFixed(0)})`,
        );
      } catch (err) {
        console.warn("Focus failed", err);
      }
    },
    [nativeTapFocus],
  );

  // ── 11. PHOTO CAPTURE ───────────────────────────────────────
  const [lastPhoto, setLastPhoto] = useState<string | null>(null);
  const [capturing, setCapturing] = useState(false);

  /**
   * photoOutput.capturePhotoToFile(settings, callbacks)
   *  Saves the photo directly to a temporary file → { filePath }.
   *
   *  settings.flash: 'on' | 'off' | 'auto'
   *  settings.enableAutoRedEyeReduction: boolean
   *  settings.enableAutoStabilization: boolean
   *
   *  callbacks.onShutter: () => void — fires when the shutter clicks
   */
  const capturePhoto = useCallback(async () => {
    if (!photoOutput || capturing) return;
    setCapturing(true);
    try {
      const { filePath } = await photoOutput.capturePhotoToFile(
        {
          flash: "auto",
          enableAutoRedEyeReduction: true,
          enableAutoStabilization: true,
        },
        {
          onShutter: () => console.log("📸 Shutter fired!"),
        },
      );
      setLastPhoto(filePath);
      console.log("Photo saved to:", filePath);
    } catch (err) {
      Alert.alert("Photo Error", String(err));
    } finally {
      setCapturing(false);
    }
  }, [photoOutput, capturing]);

  // ── 12. VIDEO RECORDING ─────────────────────────────────────
  const [recording, setRecording] = useState(false);
  const recorderRef = useRef<Awaited<
    ReturnType<typeof videoOutput.createRecorder>
  > | null>(null);

  /**
   * VIDEO RECORDING FLOW:
   *
   *  Step 1: videoOutput.createRecorder(settings) → Recorder
   *    RecorderSettings:
   *      fileNamePrefix: string
   *
   *  Step 2: recorder.startRecording(onFinished, onError)
   *    onFinished: (path: string) => void
   *    onError:    (error: CameraError) => void
   *
   *  Step 3: recorder.stopRecording()
   *    Resolves when stop is requested.
   *    onFinished fires once the file is fully written to disk.
   *
   *  ⚠️  Never reuse a Recorder — create a new one for each recording.
   */
  const startRecording = useCallback(async () => {
    if (!videoOutput || recording) return;
    try {
      const recorder = await videoOutput.createRecorder({
        fileNamePrefix: "vision_camera_demo",
      });
      recorderRef.current = recorder;

      await recorder.startRecording(
        (path) => {
          console.log("🎥 Recording finished:", path);
          Alert.alert("Recording saved", path);
          setRecording(false);
        },
        (error) => {
          console.error("Recording error:", error);
          setRecording(false);
        },
      );
      setRecording(true);
    } catch (err) {
      Alert.alert("Recording Error", String(err));
    }
  }, [videoOutput, recording]);

  const stopRecording = useCallback(async () => {
    if (!recorderRef.current) return;
    try {
      await recorderRef.current.stopRecording();
      recorderRef.current = null;
    } catch (err) {
      Alert.alert("Stop Error", String(err));
    }
  }, []);

  // ── 13. Active outputs array ─────────────────────────────────
  const activeOutputs = [photoOutput, videoOutput];

  // ── 14. Device info helpers ──────────────────────────────────
  const deviceInfo = device
    ? {
        minZoom: device.minZoom.toFixed(2),
        maxZoom: device.maxZoom.toFixed(2),
        fpsRanges: device.supportedFPSRanges?.length
          ? device.supportedFPSRanges
              .map((r) => `${r.minFps}–${r.maxFps}`)
              .join(", ")
          : "n/a",
        supportsPhotoHDR: String(device.supportsPhotoHDR),
        supportsVideoHDR: String(device.supportsVideoHDR),
        isVirtual: String(device.isVirtualDevice),
        physicalTypes:
          device.physicalDevices?.map((d) => d.type).join(", ") ?? "n/a",
        totalDevices: allDevices.length,
      }
    : null;

  // ────────────────────────────────────────────────────────────
  // RENDER
  // ────────────────────────────────────────────────────────────
  if (!hasCamPerm) {
    return (
      <View style={styles.centered}>
        <Text style={styles.permText}>Camera permission required.</Text>
        <TouchableOpacity style={styles.btn} onPress={requestCamPerm}>
          <Text style={styles.btnText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (!device) {
    return (
      <View style={styles.centered}>
        <Text style={styles.permText}>No camera device found.</Text>
      </View>
    );
  }

  return (
    <View style={styles.root}>
      {/* ── Camera Preview ───────────────────────────────── */}
      {/**
       * <Camera /> Props Reference:
       *
       *  REQUIRED:
       *   device        — CameraDevice | 'back' | 'front'
       *   isActive      — boolean; false pauses the camera (keeps it in memory)
       *
       *  OUTPUTS:
       *   outputs       — CameraOutput[]; photo and video outputs
       *
       *  VISUAL:
       *   style         — ViewStyle
       *   resizeMode    — 'cover' | 'contain'
       *   preview       — boolean (default true); show the live preview
       *
       *  ZOOM / EXPOSURE:
       *   zoom          — number | SharedValue<number>
       *   exposure      — number | SharedValue<number>  (EV bias, -3..+3)
       *
       *  CONSTRAINTS:
       *   constraints   — Constraint[]; e.g. [{ fps: 60 }]
       *
       *  ORIENTATION:
       *   orientation   — 'portrait' | 'portraitUpsideDown' |
       *                   'landscapeLeft' | 'landscapeRight'
       *
       *  NATIVE GESTURES:
       *   enableNativeZoomGesture       — boolean
       *   enableNativeTapToFocusGesture — boolean
       *
       *  REF:
       *   ref           — React.RefObject<CameraRef>
       *
       *  LIFECYCLE CALLBACKS:
       *   onConfigured  — fires when the session is configured
       *   onStarted     — fires when the camera starts streaming
       *   onStopped     — fires when the camera stops
       *   onError       — (error: CameraError) => void
       */}
      <Camera
        ref={cameraRef}
        style={styles.camera}
        device={device}
        isActive={true}
        outputs={activeOutputs}
        zoom={zoom}
        exposure={exposure}
        resizeMode={resizeMode}
        orientation={orientation}
        constraints={[{ fps: targetFps }]}
        enableNativeZoomGesture={nativeZoom}
        enableNativeTapToFocusGesture={nativeTapFocus}
        onTouchEnd={handleManualTapFocus}
        onConfigured={() => console.log("✅ Camera configured")}
        onStarted={() => console.log("▶️ Camera started")}
        onStopped={() => console.log("⏹️ Camera stopped")}
        onError={(err) => console.error("🚨 Camera error:", err)}
      />

      {/* Overlay HUD */}
      <View style={styles.hud}>
        <Text style={styles.hudText}>
          Zoom: {zoomDisplay.toFixed(1)}x | EV: {exposure > 0 ? "+" : ""}
          {exposure} | {targetFps}fps
        </Text>
        {lastPhoto && (
          <Text style={styles.hudText} numberOfLines={1}>
            📷 {lastPhoto.split("/").pop()}
          </Text>
        )}
        {recording && <Text style={styles.recBadge}>● REC</Text>}
      </View>

      {/* ── Controls ─────────────────────────────────────── */}
      <ScrollView
        style={styles.controls}
        contentContainerStyle={styles.controlsInner}
      >
        {/* Device info */}
        {deviceInfo && (
          <Section title="📱 Device Info">
            <InfoRow label="Position" value={position} />
            <InfoRow
              label="Min/Max Zoom"
              value={`${deviceInfo.minZoom}x — ${deviceInfo.maxZoom}x`}
            />
            <InfoRow label="FPS Ranges" value={deviceInfo.fpsRanges} />
            <InfoRow label="Photo HDR" value={deviceInfo.supportsPhotoHDR} />
            <InfoRow label="Video HDR" value={deviceInfo.supportsVideoHDR} />
            <InfoRow label="Virtual Device" value={deviceInfo.isVirtual} />
            <InfoRow label="Physical Lenses" value={deviceInfo.physicalTypes} />
            <InfoRow
              label="Total Cameras"
              value={String(deviceInfo.totalDevices)}
            />
          </Section>
        )}

        {/* Camera position */}
        <Section title="🔄 Camera Position">
          <Row>
            <Chip
              label="Back"
              active={position === "back"}
              onPress={() => setPosition("back")}
            />
            <Chip
              label="Front"
              active={position === "front"}
              onPress={() => setPosition("front")}
            />
          </Row>
        </Section>

        <Section title="🎨 Skia Preview">
          <TouchableOpacity
            style={styles.actionBtn}
            onPress={() => router.push("/skia-camera")}
          >
            <Text style={styles.actionBtnText}>Open Skia Camera Screen</Text>
          </TouchableOpacity>
          <Text style={styles.hint}>
            Use the SkiaCamera view for GPU-accelerated custom rendering.
          </Text>
        </Section>

        {/* Zoom */}
        <Section title={`🔍 Zoom  (${zoomDisplay.toFixed(2)}x)`}>
          <Row>
            {[0.5, 1, 2, 3, 5].map((z) => (
              <Chip
                key={z}
                label={`${z}x`}
                active={Math.abs(zoomDisplay - z) < 0.05}
                onPress={() => updateZoom(z)}
              />
            ))}
          </Row>
        </Section>

        {/* Exposure */}
        <Section title={`☀️ Exposure Bias  (${exposure} EV)`}>
          <Row>
            {[-3, -2, -1, 0, 1, 2, 3].map((ev) => (
              <Chip
                key={ev}
                label={ev > 0 ? `+${ev}` : String(ev)}
                active={exposure === ev}
                onPress={() => setExposure(ev)}
              />
            ))}
          </Row>
        </Section>

        {/* FPS */}
        <Section title="⚡ FPS Constraint">
          <Row>
            {([30, 60, 120] as FPS[]).map((fps) => (
              <Chip
                key={fps}
                label={`${fps} fps`}
                active={targetFps === fps}
                onPress={() => setTargetFps(fps)}
              />
            ))}
          </Row>
        </Section>

        {/* Resize Mode */}
        <Section title="📐 Resize Mode">
          <Row>
            <Chip
              label="Cover"
              active={resizeMode === "cover"}
              onPress={() => setResizeMode("cover")}
            />
            <Chip
              label="Contain"
              active={resizeMode === "contain"}
              onPress={() => setResizeMode("contain")}
            />
          </Row>
        </Section>

        {/* Orientation */}
        <Section title="🔁 Orientation">
          <Row>
            <Chip
              label="Portrait"
              active={orientation === "portrait"}
              onPress={() => setOrientation("portrait")}
            />
            <Chip
              label="Landscape"
              active={orientation === "landscapeLeft"}
              onPress={() => setOrientation("landscapeLeft")}
            />
          </Row>
        </Section>

        {/* Native Gestures */}
        <Section title="👆 Native Gestures">
          <Row>
            <Chip
              label={`Pinch Zoom: ${nativeZoom ? "ON" : "OFF"}`}
              active={nativeZoom}
              onPress={() => setNativeZoom((v) => !v)}
            />
            <Chip
              label={`Tap Focus: ${nativeTapFocus ? "ON" : "OFF"}`}
              active={nativeTapFocus}
              onPress={() => setNativeTapFocus((v) => !v)}
            />
          </Row>
          {!nativeTapFocus && (
            <Text style={styles.hint}>
              Tap anywhere on the preview to focus manually via
              CameraRef.focusTo()
            </Text>
          )}
        </Section>

        {/* Photo Capture */}
        <Section title="📸 Photo Capture">
          <TouchableOpacity
            style={[styles.actionBtn, capturing && styles.disabledBtn]}
            onPress={capturePhoto}
            disabled={capturing}
          >
            <Text style={styles.actionBtnText}>
              {capturing ? "Capturing…" : "Capture Photo (capturePhotoToFile)"}
            </Text>
          </TouchableOpacity>
          <Text style={styles.hint}>
            flash: auto · enableAutoRedEyeReduction · enableAutoStabilization ·
            onShutter callback
          </Text>
        </Section>

        {/* Video Recording */}
        <Section title="🎥 Video Recording">
          {!recording ? (
            <TouchableOpacity style={styles.actionBtn} onPress={startRecording}>
              <Text style={styles.actionBtnText}>Start Recording</Text>
            </TouchableOpacity>
          ) : (
            <TouchableOpacity
              style={[styles.actionBtn, styles.stopBtn]}
              onPress={stopRecording}
            >
              <Text style={styles.actionBtnText}>Stop Recording</Text>
            </TouchableOpacity>
          )}
          <Text style={styles.hint}>
            createRecorder() → startRecording(onFinished, onError) →
            stopRecording(). enableAudio={String(hasMicPerm)} · fileType: mp4
          </Text>
        </Section>
      </ScrollView>
    </View>
  );
}

// ─────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────
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

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.infoRow}>
      <Text style={styles.infoLabel}>{label}</Text>
      <Text style={styles.infoValue}>{value}</Text>
    </View>
  );
}

// ─────────────────────────────────────────────
// Styles
// ─────────────────────────────────────────────
const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: "#0a0a0a" },

  camera: { height: 320, width: "100%" },

  hud: {
    position: "absolute",
    top: 12,
    left: 12,
    right: 12,
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
  },
  hudText: {
    color: "#fff",
    fontSize: 11,
    backgroundColor: "rgba(0,0,0,0.55)",
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
    overflow: "hidden",
  },
  recBadge: {
    color: "#ff3b30",
    fontWeight: "700",
    fontSize: 11,
    backgroundColor: "rgba(0,0,0,0.55)",
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
    overflow: "hidden",
  },

  controls: { flex: 1 },
  controlsInner: { padding: 14, gap: 4 },

  section: {
    marginBottom: 16,
    backgroundColor: "#161616",
    borderRadius: 12,
    padding: 14,
    gap: 8,
  },
  sectionTitle: {
    color: "#e0e0e0",
    fontWeight: "700",
    fontSize: 13,
    marginBottom: 4,
  },

  row: { flexDirection: "row", flexWrap: "wrap", gap: 8 },

  chip: {
    paddingHorizontal: 12,
    paddingVertical: 7,
    borderRadius: 20,
    backgroundColor: "#2a2a2a",
    borderWidth: 1,
    borderColor: "#3a3a3a",
  },
  chipActive: {
    backgroundColor: "#0a84ff",
    borderColor: "#0a84ff",
  },
  chipText: { color: "#aaa", fontSize: 12, fontWeight: "500" },
  chipTextActive: { color: "#fff" },

  infoRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    paddingVertical: 2,
  },
  infoLabel: { color: "#888", fontSize: 12 },
  infoValue: {
    color: "#ddd",
    fontSize: 12,
    fontWeight: "500",
    maxWidth: "60%",
  },

  actionBtn: {
    backgroundColor: "#0a84ff",
    borderRadius: 10,
    paddingVertical: 12,
    alignItems: "center",
  },
  stopBtn: { backgroundColor: "#ff3b30" },
  disabledBtn: { opacity: 0.5 },
  actionBtnText: { color: "#fff", fontWeight: "700", fontSize: 14 },

  hint: {
    color: "#555",
    fontSize: 11,
    lineHeight: 16,
    marginTop: 4,
  },

  centered: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#0a0a0a",
  },
  permText: { color: "#ddd", fontSize: 16, marginBottom: 16 },
  btn: {
    backgroundColor: "#0a84ff",
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  btnText: { color: "#fff", fontWeight: "700" },
});
