<script setup lang="ts">
import { onMounted, ref } from "vue";
import { getHealth } from "./services/api";

const health = ref<string>("checking backend...");

onMounted(async () => {
  try {
    health.value = JSON.stringify(await getHealth());
  } catch (err) {
    health.value = `backend unreachable: ${String(err)}`;
  }
});
</script>

<template>
  <main>
    <h1>ScanDetection (M1 skeleton)</h1>
    <p>backend /api/v1/health → {{ health }}</p>
  </main>
</template>
