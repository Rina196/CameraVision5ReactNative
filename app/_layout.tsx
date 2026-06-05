import {
  DarkTheme,
  DefaultTheme,
  ThemeProvider,
} from "@react-navigation/native";
import { Stack } from "expo-router";
import { StatusBar } from "expo-status-bar";
import { useEffect } from "react";
import "react-native-reanimated";

import { useColorScheme } from "@/hooks/use-color-scheme";
import { initExecutorch } from "react-native-executorch";
import { ExpoResourceFetcher } from "react-native-executorch-expo-resource-fetcher";

export const unstable_settings = {
  anchor: "(tabs)",
};

export default function RootLayout() {
  const colorScheme = useColorScheme();

  useEffect(() => {
    initExecutorch({ resourceFetcher: ExpoResourceFetcher });
  }, []);

  return (
    <ThemeProvider value={colorScheme === "dark" ? DarkTheme : DefaultTheme}>
      <Stack>
        <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
        <Stack.Screen
          name="object-detection"
          options={{
            title: "Object Detection",
            headerBackTitle: "Explore",
          }}
        />
        <Stack.Screen
          name="frame-output-lab"
          options={{
            title: "Frame Output Lab",
            headerBackTitle: "Explore",
          }}
        />
        <Stack.Screen
          name="skia-camera"
          options={{
            title: "Skia Camera",
            headerBackTitle: "Explore",
          }}
        />
        <Stack.Screen
          name="face-mesh-camera"
          options={{
            title: "Face Mesh Camera",
            headerBackTitle: "Explore",
          }}
        />
        <Stack.Screen
          name="demo"
          options={{
            title: "Demo",
            headerBackTitle: "Explore",
          }}
        />
        <Stack.Screen
          name="modal"
          options={{ presentation: "modal", title: "Modal" }}
        />
      </Stack>
      <StatusBar style="auto" />
    </ThemeProvider>
  );
}
