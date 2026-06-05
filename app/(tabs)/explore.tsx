import { router } from "expo-router";
import React from "react";
import {
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";

const FEATURES = [
  {
    title: "Object Detection",
    route: "/object-detection",
  },
  {
    title: "Frame Output Lab",
    route: "/frame-output-lab",
  },
  {
    title: "Skia Camera",
    route: "/skia-camera",
  },
  {
    title: "Face Mesh Camera",
    route: "/face-mesh-camera",
  },
] as const;

export default function ExploreScreen() {
  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.content}
      showsVerticalScrollIndicator={false}
    >
      <View style={styles.cardList}>
        {FEATURES.map((feature) => (
          <TouchableOpacity
            key={feature.title}
            style={styles.card}
            activeOpacity={0.9}
            onPress={() => router.push(feature.route)}
          >
            <Text style={styles.cardTitle}>{feature.title}</Text>
          </TouchableOpacity>
        ))}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#08111f",
  },
  content: {
    padding: 16,
    paddingBottom: 28,
  },
  cardList: {
    gap: 14,
    paddingTop: 8,
  },
  card: {
    borderRadius: 24,
    backgroundColor: "#122239",
    paddingHorizontal: 18,
    paddingVertical: 24,
    borderWidth: 1,
    borderColor: "#1f3a5a",
  },
  cardTitle: {
    color: "#f6fbff",
    fontSize: 24,
    fontWeight: "800",
  },
});
