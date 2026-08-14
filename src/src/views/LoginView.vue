<script setup lang="ts">
/** 登录视图（§T3）：极简 ZINE 风格，与 RailNav 品牌一致。 */
import { ref } from "vue";
import { useAuth } from "../composables/useAuth";

const auth = useAuth();
const username = ref("");
const password = ref("");
const error = ref<string | null>(null);
const busy = ref(false);

async function submit() {
  error.value = null;
  busy.value = true;
  try {
    await auth.login(username.value.trim(), password.value);
  } catch (e: unknown) {
    const err = e as { status?: number; message?: string };
    error.value =
      err?.status === 401
        ? "用户名或密码错误"
        : err?.message || "登录失败，请稍后重试";
  } finally {
    busy.value = false;
    password.value = "";
  }
}
</script>

<template>
  <div class="login-wrap">
    <form class="login-card" @submit.prevent="submit">
      <div class="brand"><span class="mark" aria-hidden="true"></span> 射线评片</div>
      <p class="sub">焊缝缺陷智能检测 · NB/T47013.2-2015</p>

      <label class="field">
        <span>用户名</span>
        <input v-model="username" type="text" autocomplete="username" :disabled="busy" />
      </label>
      <label class="field">
        <span>密码</span>
        <input
          v-model="password"
          type="password"
          autocomplete="current-password"
          :disabled="busy"
        />
      </label>

      <p v-if="error" class="err">{{ error }}</p>

      <button type="submit" :disabled="busy || !username || !password">
        {{ busy ? "登录中…" : "登 录" }}
      </button>
    </form>
  </div>
</template>

<style scoped>
.login-wrap {
  display: grid;
  place-items: center;
  min-height: 100vh;
  background: #f4f2ef;
}
.login-card {
  width: 340px;
  background: #fff;
  border: 1px solid #e3ddd5;
  border-radius: 10px;
  padding: 28px 26px 30px;
  box-shadow: 0 8px 30px rgba(20, 30, 50, 0.08);
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.brand {
  display: flex;
  align-items: center;
  gap: 10px;
  font-family: Georgia, "Times New Roman", serif;
  font-size: 22px;
  font-weight: 700;
  color: #14233f;
}
.mark {
  width: 14px;
  height: 14px;
  background: #2f6bff;
  display: inline-block;
  border-radius: 2px;
}
.sub {
  margin: -8px 0 6px;
  font-size: 12px;
  color: #8a8378;
  letter-spacing: 0.02em;
}
.field {
  display: flex;
  flex-direction: column;
  gap: 6px;
  font-size: 12px;
  color: #4a443c;
}
.field input {
  height: 38px;
  padding: 0 12px;
  border: 1px solid #d8d1c8;
  border-radius: 6px;
  font-size: 14px;
  outline: none;
}
.field input:focus {
  border-color: #2f6bff;
}
.err {
  color: #c0392b;
  font-size: 12px;
  margin: 0;
}
button {
  height: 40px;
  border: none;
  border-radius: 6px;
  background: #2f6bff;
  color: #fff;
  font-size: 14px;
  letter-spacing: 0.2em;
  cursor: pointer;
}
button:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}
</style>
