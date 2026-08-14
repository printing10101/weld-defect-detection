<script setup lang="ts">
/** 三步旅程指示器：01 上传 → 02 处理 → 03 报告（步骤由真实旅程阶段驱动）。 */
defineProps<{ current: 1 | 2 | 3 }>();

const STEPS = ["上传", "处理", "报告"] as const;
</script>

<template>
  <div class="stepper" aria-label="检测旅程步骤">
    <template v-for="(label, i) in STEPS" :key="label">
      <div v-if="i > 0" class="bar" :class="{ done: current > i }" />
      <div class="step" :class="{ active: current === i + 1, done: current > i + 1 }">
        <span class="num">{{ `0${i + 1}` }}</span>{{ label }}
      </div>
    </template>
  </div>
</template>
