<script setup lang="ts">
/** 应用壳（DESIGN.md：AppShell = RailNav + main 主区，托管旅程/档案/批量/设备四视图）。 */
import { ref } from "vue";
import type { ViewId } from "../types/api";
import RailNav from "./RailNav.vue";
import JourneyView from "../views/JourneyView.vue";
import ArchiveView from "../views/ArchiveView.vue";
import BatchView from "../views/BatchView.vue";
import DeviceView from "../views/DeviceView.vue";

const view = ref<ViewId>("journey");
</script>

<template>
  <div class="app-shell">
    <RailNav :active="view" @navigate="view = $event" />
    <main class="main">
      <JourneyView v-show="view === 'journey'" @archive="view = 'archive'" />
      <ArchiveView v-show="view === 'archive'" :active="view === 'archive'" />
      <BatchView v-show="view === 'batch'" :active="view === 'batch'" @archive="view = 'archive'" />
      <DeviceView v-show="view === 'device'" />
    </main>
  </div>
</template>
