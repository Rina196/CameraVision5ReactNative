// =========================================
// INSTALL SLIDER
// =========================================

// yarn add @react-native-community/slider

// =========================================
// IMPORTS
// =========================================

import Slider from "@react-native-community/slider";
import { ImageFormat, Skia } from "@shopify/react-native-skia";
import * as FileSystem from "expo-file-system/legacy";
import * as MediaLibrary from "expo-media-library";

import React, { useCallback, useEffect, useRef, useState } from "react";

import {
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";

import {
  type TargetVideoPixelFormat,
  useCameraDevice,
  useCameraPermission,
} from "react-native-vision-camera";
import { createSynchronizable } from "react-native-worklets";

import {
  SkiaCamera,
  type SkiaCameraProps,
  type SkiaCameraRef,
  type SkiaOnFrameState,
} from "react-native-vision-camera-skia";

// =========================================
// TYPES
// =========================================

type CameraPosition = "back" | "front";
type AdjustmentKey =
  | "brightness"
  | "contrast"
  | "saturation"
  | "warmth"
  | "blueTone"
  | "whiteBalance";
type EditorMode = "adjust" | "filters";
type FilterType =
  | "normal"
  | "grayscale"
  | "bright"
  | "blur"
  | "warm"
  | "cool"
  | "beauty"
  | "cartoon"
  | "emoji"
  | "tracking";

const SKIA_PIXEL_FORMAT: TargetVideoPixelFormat = "rgb";
const CLAMP_TILE_MODE = 0;
const cameraPositionSync = createSynchronizable<CameraPosition>("back");
const brightnessSync = createSynchronizable(0);
const contrastSync = createSynchronizable(1);
const saturationSync = createSynchronizable(1);
const warmthSync = createSynchronizable(0);
const blueToneSync = createSynchronizable(0);
const whiteBalanceSync = createSynchronizable(0);
const selectedFilterSync = createSynchronizable<FilterType>("normal");
const FILTERS: FilterType[] = [
  "normal",
  "grayscale",
  "bright",
  "blur",
  "warm",
  "cool",
  "beauty",
  "cartoon",
  "emoji",
  "tracking",
];
const ADJUSTMENT_OPTIONS: {
  key: AdjustmentKey;
  label: string;
  icon: string;
}[] = [
  { key: "brightness", label: "Shadows", icon: "◑" },
  { key: "contrast", label: "Contrast", icon: "◐" },
  { key: "saturation", label: "Saturation", icon: "◍" },
  { key: "warmth", label: "Warmth", icon: "☼" },
  { key: "blueTone", label: "Cool", icon: "◌" },
  { key: "whiteBalance", label: "Balance", icon: "◎" },
];

// =========================================
// COMPONENT
// =========================================

export default function CameraEditorScreen() {
  const { hasPermission, requestPermission } = useCameraPermission();
  const cameraRef = useRef<SkiaCameraRef>(null);

  const [cameraPosition, setCameraPosition] = useState<CameraPosition>("back");

  // =========================================
  // LIVE EDIT CONTROLS
  // =========================================

  const [brightness, setBrightness] = useState(0);

  const [contrast, setContrast] = useState(1);

  const [saturation, setSaturation] = useState(1);

  const [warmth, setWarmth] = useState(0);

  const [blueTone, setBlueTone] = useState(0);

  const [whiteBalance, setWhiteBalance] = useState(0);
  const [activeAdjustment, setActiveAdjustment] =
    useState<AdjustmentKey>("brightness");
  const [editorMode, setEditorMode] = useState<EditorMode>("adjust");
  const [selectedFilter, setSelectedFilter] = useState<FilterType>("normal");
  const [isSaving, setIsSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);

  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission]);

  useEffect(() => {
    cameraPositionSync.setBlocking(cameraPosition);
  }, [cameraPosition]);

  useEffect(() => {
    brightnessSync.setBlocking(brightness);
  }, [brightness]);

  useEffect(() => {
    contrastSync.setBlocking(contrast);
  }, [contrast]);

  useEffect(() => {
    saturationSync.setBlocking(saturation);
  }, [saturation]);

  useEffect(() => {
    warmthSync.setBlocking(warmth);
  }, [warmth]);

  useEffect(() => {
    blueToneSync.setBlocking(blueTone);
  }, [blueTone]);

  useEffect(() => {
    whiteBalanceSync.setBlocking(whiteBalance);
  }, [whiteBalance]);

  useEffect(() => {
    selectedFilterSync.setBlocking(selectedFilter);
  }, [selectedFilter]);

  const device = useCameraDevice(cameraPosition);

  const captureFilteredPhoto = useCallback(async () => {
    try {
      setIsSaving(true);
      const permission = await MediaLibrary.requestPermissionsAsync();
      if (!permission.granted) {
        setSaveMessage("Permission denied");
        return;
      }
      const snapshot = cameraRef.current?.takeSnapshot();
      if (!snapshot) {
        setSaveMessage("Snapshot not ready");
        return;
      }
      const base64 = snapshot.encodeToBase64(ImageFormat.JPEG, 95);
      const fileUri = FileSystem.cacheDirectory + `photo-${Date.now()}.jpg`;
      await FileSystem.writeAsStringAsync(fileUri, base64, {
        encoding: FileSystem.EncodingType.Base64,
      });
      await MediaLibrary.saveToLibraryAsync(fileUri);
      setSaveMessage("Photo saved!");
    } catch (e) {
      console.log("SAVE ERROR => ", e);
      setSaveMessage("Save failed");
    } finally {
      setIsSaving(false);
    }
  }, []);

  const adjustmentConfig: Record<
    AdjustmentKey,
    {
      label: string;
      min: number;
      max: number;
      value: number;
      setValue: (value: number) => void;
    }
  > = {
    brightness: {
      label: "Shadows",
      min: -40,
      max: 40,
      value: brightness,
      setValue: setBrightness,
    },
    contrast: {
      label: "Contrast",
      min: 0.5,
      max: 2,
      value: contrast,
      setValue: setContrast,
    },
    saturation: {
      label: "Saturation",
      min: 0,
      max: 2,
      value: saturation,
      setValue: setSaturation,
    },
    warmth: {
      label: "Warmth",
      min: -0.5,
      max: 0.5,
      value: warmth,
      setValue: setWarmth,
    },
    blueTone: {
      label: "Cool",
      min: -0.5,
      max: 0.5,
      value: blueTone,
      setValue: setBlueTone,
    },
    whiteBalance: {
      label: "Balance",
      min: -20,
      max: 20,
      value: whiteBalance,
      setValue: setWhiteBalance,
    },
  };
  const activeControl = adjustmentConfig[activeAdjustment];

  // =========================================
  // DRAW FRAME
  // =========================================

  function drawFrame({ frameTexture, canvas }: SkiaOnFrameState) {
    "worklet";

    const paint = Skia.Paint();
    let shouldApplyBrightOverlay = false;
    let shouldApplyWarmOverlay = false;
    let shouldApplyCoolOverlay = false;
    let shouldApplyBeautyOverlay = false;
    let shouldApplyCartoonOverlay = false;

    // =========================================
    // BRIGHTNESS
    // =========================================

    const brightnessValue = brightnessSync.getDirty();
    const contrastValue = contrastSync.getDirty();
    const saturationValue = saturationSync.getDirty();
    const warmthValue = warmthSync.getDirty();
    const blueToneValue = blueToneSync.getDirty();
    const whiteBalanceValue = whiteBalanceSync.getDirty();
    const selectedFilterValue = selectedFilterSync.getDirty();
    const cameraPositionValue = cameraPositionSync.getDirty();

    const shadowAmount = brightnessValue / 40;
    const shadowOffset = brightnessValue * 0.22;

    // =========================================
    // CONTRAST
    // =========================================

    const shadowContrastAdjustment =
      shadowAmount > 0 ? 1 - shadowAmount * 0.18 : 1 - shadowAmount * 0.1;
    const c = contrastValue * shadowContrastAdjustment;

    // =========================================
    // SATURATION
    // =========================================

    const s = saturationValue;

    // =========================================
    // WARMTH
    // =========================================

    const w = warmthValue;

    // =========================================
    // BLUE TONE
    // =========================================

    const bt = blueToneValue;

    // =========================================
    // WHITE BALANCE
    // =========================================

    const wb = whiteBalanceValue / 20;
    const redBalanceGain = 1 + wb * 0.18;
    const greenBalanceGain = 1 - Math.abs(wb) * 0.05;
    const blueBalanceGain = 1 - wb * 0.18;

    // =========================================
    // COMBINED COLOR MATRIX
    // =========================================

    const matrix = [
      // RED
      (c * (0.213 + 0.787 * s) + w) * redBalanceGain,
      c * (0.715 - 0.715 * s),
      c * (0.072 - 0.072 * s) + bt,
      0,
      shadowOffset,

      // GREEN
      c * (0.213 - 0.213 * s),
      c * (0.715 + 0.285 * s) * greenBalanceGain,
      c * (0.072 - 0.072 * s),
      0,
      shadowOffset,

      // BLUE
      c * (0.213 - 0.213 * s),
      c * (0.715 - 0.715 * s),
      (c * (0.072 + 0.928 * s) + bt) * blueBalanceGain,
      0,
      shadowOffset,

      // ALPHA
      0,
      0,
      0,
      1,
      0,
    ];

    paint.setColorFilter(Skia.ColorFilter.MakeMatrix(matrix));

    // =========================================
    // FILTERS
    // =========================================

    if (selectedFilterValue === "grayscale") {
      paint.setColorFilter(
        Skia.ColorFilter.MakeMatrix([
          0.33, 0.33, 0.33, 0, 0, 0.33, 0.33, 0.33, 0, 0, 0.33, 0.33, 0.33, 0,
          0, 0, 0, 0, 1, 0,
        ]),
      );
    }

    if (selectedFilterValue === "bright") {
      shouldApplyBrightOverlay = true;
    }

    if (selectedFilterValue === "blur") {
      paint.setImageFilter(Skia.ImageFilter.MakeBlur(12, 12, CLAMP_TILE_MODE));
    }

    if (selectedFilterValue === "warm") {
      shouldApplyWarmOverlay = true;
    }

    if (selectedFilterValue === "cool") {
      shouldApplyCoolOverlay = true;
    }

    if (selectedFilterValue === "beauty") {
      shouldApplyBeautyOverlay = true;
      shouldApplyBrightOverlay = true;
    }

    if (selectedFilterValue === "cartoon") {
      paint.setImageFilter(Skia.ImageFilter.MakeBlur(1, 1, CLAMP_TILE_MODE));
      shouldApplyCartoonOverlay = true;
    }

    // =========================================
    // FIX FRONT CAMERA MIRROR
    // =========================================

    canvas.save();

    if (cameraPositionValue === "front") {
      canvas.scale(-1, 1);

      canvas.translate(-frameTexture.width(), 0);
    }

    // =========================================
    // DRAW CAMERA
    // =========================================

    canvas.drawImage(frameTexture, 0, 0, paint);

    canvas.restore();

    if (shouldApplyBrightOverlay) {
      const brightOverlayPaint = Skia.Paint();
      brightOverlayPaint.setColor(Skia.Color("rgba(255,255,255,0.12)"));
      canvas.drawRect(
        Skia.XYWHRect(0, 0, frameTexture.width(), frameTexture.height()),
        brightOverlayPaint,
      );
    }

    if (shouldApplyWarmOverlay) {
      const warmOverlayPaint = Skia.Paint();
      warmOverlayPaint.setColor(Skia.Color("rgba(255,210,120,0.14)"));
      canvas.drawRect(
        Skia.XYWHRect(0, 0, frameTexture.width(), frameTexture.height()),
        warmOverlayPaint,
      );
    }

    if (shouldApplyCoolOverlay) {
      const coolOverlayPaint = Skia.Paint();
      coolOverlayPaint.setColor(Skia.Color("rgba(120,180,255,0.12)"));
      canvas.drawRect(
        Skia.XYWHRect(0, 0, frameTexture.width(), frameTexture.height()),
        coolOverlayPaint,
      );
    }

    if (shouldApplyBeautyOverlay) {
      const beautyOverlayPaint = Skia.Paint();
      beautyOverlayPaint.setColor(Skia.Color("rgba(255,228,214,0.12)"));
      canvas.drawRect(
        Skia.XYWHRect(0, 0, frameTexture.width(), frameTexture.height()),
        beautyOverlayPaint,
      );

      const beautyGlowPaint = Skia.Paint();
      beautyGlowPaint.setColor(Skia.Color("rgba(255,255,255,0.06)"));
      canvas.drawRect(
        Skia.XYWHRect(0, 0, frameTexture.width(), frameTexture.height()),
        beautyGlowPaint,
      );
    }

    if (shouldApplyCartoonOverlay) {
      const cartoonOverlayPaint = Skia.Paint();
      cartoonOverlayPaint.setColor(Skia.Color("rgba(255,255,255,0.08)"));
      canvas.drawRect(
        Skia.XYWHRect(0, 0, frameTexture.width(), frameTexture.height()),
        cartoonOverlayPaint,
      );
    }

    if (selectedFilterValue === "emoji") {
      const emojiPaint = Skia.Paint();
      emojiPaint.setColor(Skia.Color("#ffd60a"));
      canvas.drawCircle(180, 250, 36, emojiPaint);
      canvas.drawCircle(270, 250, 28, emojiPaint);
    }

    if (selectedFilterValue === "tracking") {
      const trackingPaint = Skia.Paint();
      trackingPaint.setColor(Skia.Color("#00ff99"));
      trackingPaint.setStrokeWidth(6);
      trackingPaint.setStyle(1);
      canvas.drawRect(Skia.XYWHRect(120, 220, 220, 320), trackingPaint);
    }

    // =========================================
    // UI FRAME BORDER
    // =========================================

    const borderPaint = Skia.Paint();

    borderPaint.setColor(Skia.Color("rgba(255,255,255,0.4)"));

    borderPaint.setStrokeWidth(3);

    borderPaint.setStyle(1);

    canvas.drawRect(Skia.XYWHRect(20, 60, 350, 620), borderPaint);
  }

  // =========================================
  // HANDLE FRAME
  // =========================================

  function handleFrame(
    frame: Parameters<SkiaCameraProps["onFrame"]>[0],

    render: Parameters<SkiaCameraProps["onFrame"]>[1],
  ) {
    "worklet";

    render(drawFrame);

    frame.dispose();
  }

  // =========================================
  // PERMISSION
  // =========================================

  if (!hasPermission) {
    return (
      <View style={styles.permissionContainer}>
        <Text style={styles.permissionTitle}>Camera Permission Required</Text>

        <TouchableOpacity
          style={styles.permissionButton}
          onPress={requestPermission}
        >
          <Text style={styles.permissionButtonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // =========================================
  // NO DEVICE
  // =========================================

  if (!device) {
    return (
      <View style={styles.permissionContainer}>
        <Text style={styles.permissionTitle}>No Camera Found</Text>
      </View>
    );
  }

  // =========================================
  // MAIN UI
  // =========================================

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" />

      {/* CAMERA */}
      <SkiaCamera
        ref={cameraRef}
        key={cameraPosition}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        orientationSource="interface"
        pixelFormat={SKIA_PIXEL_FORMAT}
        onFrame={handleFrame}
        photo={true}
      />

      {/* TOP BAR */}
      <View style={styles.topBar}>
        <TouchableOpacity
          style={[
            styles.topActionButton,
            isSaving && styles.topActionButtonDisabled,
          ]}
          disabled={isSaving}
          onPress={captureFilteredPhoto}
        >
          <Text style={styles.topActionButtonText}>
            {isSaving ? "Saving..." : "Capture"}
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.flipButton}
          onPress={() =>
            setCameraPosition((prev) => (prev === "back" ? "front" : "back"))
          }
        >
          <Text style={styles.flipButtonText}>Flip</Text>
        </TouchableOpacity>
      </View>

      {saveMessage ? (
        <View style={styles.saveToast}>
          <Text style={styles.saveToastText}>{saveMessage}</Text>
        </View>
      ) : null}

      {/* CONTROLS */}
      <View style={styles.controlsContainer}>
        <View style={styles.controlsHeader}>
          <TouchableOpacity style={styles.headerGhostButton}>
            <Text style={styles.headerGhostText}>Cancel</Text>
          </TouchableOpacity>

          <Text style={styles.controlsTitle}>
            {editorMode === "adjust" ? "ADJUST" : "FILTERS"}
          </Text>

          <TouchableOpacity style={styles.headerPrimaryButton}>
            <Text style={styles.headerPrimaryText}>Done</Text>
          </TouchableOpacity>
        </View>

        {editorMode === "adjust" ? (
          <>
            <View style={styles.adjustmentLabelWrap}>
              <Text style={styles.adjustmentLabel}>{activeControl.label}</Text>
              <View style={styles.adjustmentUnderline} />
            </View>

            <View style={styles.adjustmentIconsRow}>
              {ADJUSTMENT_OPTIONS.map((option) => {
                const isActive = option.key === activeAdjustment;

                return (
                  <TouchableOpacity
                    key={option.key}
                    style={[
                      styles.adjustmentIconButton,
                      isActive && styles.adjustmentIconButtonActive,
                    ]}
                    onPress={() => setActiveAdjustment(option.key)}
                  >
                    <Text
                      style={[
                        styles.adjustmentIcon,
                        isActive && styles.adjustmentIconActive,
                      ]}
                    >
                      {option.icon}
                    </Text>
                  </TouchableOpacity>
                );
              })}
            </View>

            <Slider
              style={styles.adjustmentSlider}
              value={activeControl.value}
              minimumValue={activeControl.min}
              maximumValue={activeControl.max}
              minimumTrackTintColor="#1f1f1f"
              maximumTrackTintColor="#d4d0d5"
              thumbTintColor="#f2c94c"
              onValueChange={activeControl.setValue}
            />
          </>
        ) : null}

        {editorMode === "filters" ? (
          <>
            <View style={styles.adjustmentLabelWrap}>
              <Text style={styles.adjustmentLabel}>Filters</Text>
              <View style={styles.adjustmentUnderline} />
            </View>

            <ScrollView
              horizontal
              showsHorizontalScrollIndicator={false}
              contentContainerStyle={styles.filterChipsRow}
            >
              {FILTERS.map((filter) => {
                const isActive = filter === selectedFilter;

                return (
                  <TouchableOpacity
                    key={filter}
                    style={[
                      styles.filterChip,
                      styles.filterChipLarge,
                      isActive && styles.filterChipActive,
                    ]}
                    onPress={() => setSelectedFilter(filter)}
                  >
                    <Text
                      style={[
                        styles.filterChipText,
                        isActive && styles.filterChipTextActive,
                      ]}
                    >
                      {filter.toUpperCase()}
                    </Text>
                  </TouchableOpacity>
                );
              })}
            </ScrollView>

            <Text style={styles.selectedFilterText}>
              Selected: {selectedFilter.toUpperCase()}
            </Text>
          </>
        ) : null}

        <View style={styles.editorTabsRow}>
          <TouchableOpacity
            style={
              editorMode === "adjust"
                ? styles.editorTabActive
                : styles.editorTab
            }
            onPress={() => setEditorMode("adjust")}
          >
            <Text
              style={
                editorMode === "adjust"
                  ? styles.editorTabIconActive
                  : styles.editorTabIcon
              }
            >
              ☀
            </Text>
            <Text
              style={
                editorMode === "adjust"
                  ? styles.editorTabTextActive
                  : styles.editorTabText
              }
            >
              Adjust
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={
              editorMode === "filters"
                ? styles.editorTabActive
                : styles.editorTab
            }
            onPress={() => setEditorMode("filters")}
          >
            <Text
              style={
                editorMode === "filters"
                  ? styles.editorTabIconActive
                  : styles.editorTabIcon
              }
            >
              ◌
            </Text>
            <Text
              style={
                editorMode === "filters"
                  ? styles.editorTabTextActive
                  : styles.editorTabText
              }
            >
              Filters
            </Text>
          </TouchableOpacity>
        </View>
      </View>
    </View>
  );
}

// =========================================
// STYLES
// =========================================

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#000",
  },

  topBar: {
    position: "absolute",
    top: 70,
    left: 20,
    right: 20,
    flexDirection: "row",
    justifyContent: "space-between",
    zIndex: 100,
  },

  topActionButton: {
    backgroundColor: "rgba(255,255,255,0.22)",
    paddingHorizontal: 18,
    paddingVertical: 12,
    borderRadius: 30,
  },

  topActionButtonDisabled: {
    opacity: 0.7,
  },

  topActionButtonText: {
    color: "white",
    fontWeight: "700",
  },

  flipButton: {
    backgroundColor: "rgba(255,255,255,0.2)",

    paddingHorizontal: 18,
    paddingVertical: 12,

    borderRadius: 30,
  },

  flipButtonText: {
    color: "white",
    fontWeight: "700",
  },

  saveToast: {
    position: "absolute",
    top: 126,
    left: 20,
    right: 20,
    borderRadius: 16,
    backgroundColor: "rgba(15,15,15,0.78)",
    paddingHorizontal: 16,
    paddingVertical: 12,
  },

  saveToastText: {
    color: "#fff",
    fontSize: 13,
    fontWeight: "600",
    textAlign: "center",
  },

  controlsContainer: {
    position: "absolute",
    bottom: 0,
    left: 0,
    right: 0,
    paddingTop: 18,
    paddingHorizontal: 14,
    paddingBottom: 26,
    backgroundColor: "rgba(252,250,247,0.96)",
    borderTopLeftRadius: 32,
    borderTopRightRadius: 32,
  },

  controlsHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },

  headerGhostButton: {
    borderRadius: 999,
    backgroundColor: "#d8d4d7",
    paddingHorizontal: 10,
    paddingVertical: 6,
  },

  headerGhostText: {
    color: "#151515",
    fontWeight: "700",
    fontSize: 13,
  },

  controlsTitle: {
    color: "#9a949b",
    fontSize: 28,
    fontWeight: "700",
    letterSpacing: 1.5,
  },

  headerPrimaryButton: {
    borderRadius: 999,
    backgroundColor: "#f4db7f",
    paddingHorizontal: 14,
    paddingVertical: 7,
  },

  headerPrimaryText: {
    color: "#7e6832",
    fontWeight: "700",
    fontSize: 13,
  },

  quickActionsRow: {
    marginTop: 18,
    flexDirection: "row",
    justifyContent: "space-between",
    paddingHorizontal: 8,
  },

  quickActionIcon: {
    color: "#9f9aa2",
    fontSize: 24,
    width: 40,
    textAlign: "center",
  },

  filterChipsRow: {
    paddingTop: 22,
    paddingBottom: 10,
    paddingHorizontal: 2,
    gap: 10,
  },

  filterChip: {
    minWidth: 92,
    borderRadius: 999,
    paddingVertical: 10,
    paddingHorizontal: 16,
    backgroundColor: "rgba(80,71,78,0.7)",
    alignItems: "center",
    justifyContent: "center",
  },

  filterChipLarge: {
    minWidth: 104,
  },

  filterChipActive: {
    backgroundColor: "#fff",
  },

  filterChipText: {
    color: "#fff",
    fontSize: 12,
    fontWeight: "700",
  },

  filterChipTextActive: {
    color: "#161616",
  },

  adjustmentLabelWrap: {
    alignItems: "center",
    marginTop: 2,
  },

  adjustmentLabel: {
    color: "#1d1d1d",
    fontSize: 34,
    fontWeight: "300",
    letterSpacing: 1,
  },

  adjustmentUnderline: {
    marginTop: 4,
    width: 74,
    height: 4,
    borderRadius: 999,
    backgroundColor: "#1d1d1d",
  },

  adjustmentIconsRow: {
    marginTop: 22,
    flexDirection: "row",
    justifyContent: "space-between",
    paddingHorizontal: 10,
  },

  adjustmentIconButton: {
    width: 54,
    height: 54,
    borderRadius: 27,
    borderWidth: 1.5,
    borderColor: "#beb8bf",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#fff",
  },

  adjustmentIconButtonActive: {
    borderColor: "#1f1f1f",
  },

  adjustmentIcon: {
    color: "#232323",
    fontSize: 26,
  },

  adjustmentIconActive: {
    color: "#000",
  },

  adjustmentSlider: {
    marginTop: 18,
  },

  selectedFilterText: {
    marginTop: 18,
    textAlign: "center",
    color: "#4a454b",
    fontSize: 13,
    fontWeight: "600",
  },

  editorTabsRow: {
    marginTop: 14,
    flexDirection: "row",
    justifyContent: "center",
    gap: 36,
  },

  editorTab: {
    alignItems: "center",
  },

  editorTabActive: {
    alignItems: "center",
  },

  editorTabIcon: {
    color: "#9d98a0",
    fontSize: 25,
  },

  editorTabIconActive: {
    color: "#111",
    fontSize: 25,
  },

  editorTabText: {
    marginTop: 4,
    color: "#9d98a0",
    fontSize: 13,
  },

  editorTabTextActive: {
    marginTop: 4,
    color: "#111",
    fontSize: 13,
    fontWeight: "600",
  },

  permissionContainer: {
    flex: 1,

    justifyContent: "center",

    alignItems: "center",

    backgroundColor: "#000",
  },

  permissionTitle: {
    color: "white",

    fontSize: 22,

    fontWeight: "700",
  },

  permissionButton: {
    marginTop: 20,

    backgroundColor: "#0A84FF",

    paddingHorizontal: 22,

    paddingVertical: 14,

    borderRadius: 20,
  },

  permissionButtonText: {
    color: "white",

    fontWeight: "700",
  },
});
