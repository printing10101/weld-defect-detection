<script setup lang="ts">
/** 应用壳（DESIGN.md：AppShell = RailNav + main 主区，托管旅程/档案两视图）。 */
import { ref } from "vue";
import type { ViewId } from "../types/api";
import RailNav from "./RailNav.vue";
import JourneyView from "../views/JourneyView.vue";
import ArchiveView from "../views/ArchiveView.vue";

const view = ref<ViewId>("journey");
</script>

<template>
  <div class="app-shell">
    <RailNav :active="view" @navigate="view = $event" />
    <main class="main">
      <JourneyView v-show="view === 'journey'" @archive="view = 'archive'" />
      <ArchiveView v-show="view === 'archive'" :active="view === 'archive'" />
    </main>
  </div>
</template>
