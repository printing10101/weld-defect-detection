<script setup lang="ts">
/** 左侧导航（DESIGN.md：窄 226px，含 14px 钴蓝方块 mark + 衬线应用名 + 等宽导航）。 */
import type { ViewId } from "../types/api";
import { useAuth } from "../composables/useAuth";

defineProps<{ active: ViewId }>();
const emit = defineEmits<{ navigate: [view: ViewId] }>();

const auth = useAuth();

const NAV: { id: ViewId; label: string }[] = [
  { id: "journey", label: "检测旅程" },
  { id: "batch", label: "批量检测" },
  { id: "archive", label: "档案检索" },
];
</script>

<template>
  <aside class="rail">
    <div class="brand">
      <span class="mark" aria-hidden="true"></span>
      <span class="name">射线评片</span>
    </div>
    <div class="sub">焊缝缺陷智能检测<br />NB/T47013.2-2015 · 本地优先</div>
    <nav>
      <button
        v-for="item in NAV"
        :key="item.id"
        :class="{ active: active === item.id }"
        type="button"
        @click="emit('navigate', item.id)"
      >
        <span class="dot" aria-hidden="true"></span>{{ item.label }}
      </button>
    </nav>
    <div class="foot">
      上传 → 处理 → 报告解读<br />
      极简 ZINE · 单一钴蓝锚<br />
      内容均来自真实检测数据
      <div v-if="auth.isAuthenticated.value" class="user">
        <span class="who">
          {{ auth.state.user?.display_name || auth.state.user?.username }}
          <em class="role">{{ auth.state.user?.role }}</em>
        </span>
        <button type="button" class="logout" @click="auth.logout()">退出</button>
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
.who {
  color: #e8eefc;
}
.role {
  font-style: normal;
  margin-left: 6px;
  padding: 1px 6px;
  border-radius: 3px;
  background: rgba(47, 107, 255, 0.35);
  font-size: 11px;
  letter-spacing: 0.04em;
}
.logout {
  align-self: flex-start;
  background: transparent;
  border: 1px solid rgba(255, 255, 255, 0.3);
  color: #cfd8ee;
  border-radius: 5px;
  padding: 3px 10px;
  font-size: 11px;
  cursor: pointer;
}
.logout:hover {
  background: rgba(255, 255, 255, 0.1);
}
</style>
