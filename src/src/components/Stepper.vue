<script setup lang="ts">
/** 三步旅程指示器：01 底片导入 → 02 自动评定 → 03 报告签发（步骤由真实旅程阶段驱动）。 */
defineProps<{ current: 1 | 2 | 3 }>();

const STEPS = ["底片导入", "自动评定", "报告签发"] as const;
</script>

<template>
  <div
    class="stepper"
    aria-label="评定流程步骤"
  >
    <template
      v-for="(label, i) in STEPS"
      :key="label"
    >
      <div
        v-if="i > 0"
        class="bar"
        :class="{ done: current > i }"
      />
      <div
        class="step"
        :class="{ active: current === i + 1, done: current > i + 1 }"
      >
        <span class="num">{{ `0${i + 1}` }}</span>{{ label }}
      </div>
    </template>
  </div>
</template>
