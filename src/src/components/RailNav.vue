<script setup lang="ts">
/** 左侧导航（DESIGN.md：窄 226px，含 14px 钴蓝方块 mark + 衬线应用名 + 等宽导航）。 */
import { ref } from "vue";
import type { ViewId } from "../types/api";
import { getOperatorName, setOperatorName } from "../services/operator";

defineProps<{ active: ViewId }>();
const emit = defineEmits<{ navigate: [view: ViewId] }>();

// 操作员姓名（单机无用户系统）：显示当前姓名，点击可编辑并保存到 localStorage
const operatorName = ref(getOperatorName());

function editOperatorName(): void {
  const name = window.prompt("操作员姓名（用于报告签名与审计留痕）", operatorName.value);
  if (name === null) return; // 取消
  setOperatorName(name);
  operatorName.value = getOperatorName();
}

const NAV: { id: ViewId; label: string }[] = [
  { id: "journey", label: "检测旅程" },
  { id: "batch", label: "批量检测" },
  { id: "archive", label: "档案检索" },
  { id: "device", label: "设备标定" },
];
</script>

<template>
  <aside class="rail">
    <div class="brand">
      <span
        class="mark"
        aria-hidden="true"
      />
      <span class="name">射线评片</span>
    </div>
    <div class="sub">
      焊缝缺陷智能检测<br>NB/T47013.2-2015 · 本地优先
    </div>
    <nav>
      <button
        v-for="item in NAV"
        :key="item.id"
        :class="{ active: active === item.id }"
        type="button"
        @click="emit('navigate', item.id)"
      >
        <span
          class="dot"
          aria-hidden="true"
        />{{ item.label }}
      </button>
    </nav>
    <div class="foot">
      上传 → 处理 → 报告解读<br>
      极简 ZINE · 单一钴蓝锚<br>
      内容均来自真实检测数据
      <div class="user">
        <button
          type="button"
          class="operator"
          :title="`操作员：${operatorName}（点击修改）`"
          @click="editOperatorName"
        >
          操作员：{{ operatorName }}
        </button>
      </div>
    </div>
  </aside>
</template>

<style scoped>
.user {
  margin-top: 12px;
  padding-top: 10px;
  border-top: 1px solid rgba(255, 255, 255, 0.12);
  display: flex;
  flex-direction: column;
  gap: 6px;
  font-size: 12px;
}
.operator {
  align-self: flex-start;
  background: transparent;
  border: 1px solid rgba(255, 255, 255, 0.3);
  color: #cfd8ee;
  border-radius: 5px;
  padding: 3px 10px;
  font-size: 11px;
  cursor: pointer;
}
.operator:hover {
  background: rgba(255, 255, 255, 0.1);
}
</style>
