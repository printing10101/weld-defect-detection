<script setup lang="ts">
/** 登录页（C-06/C-07 三员身份）：用户名 + SM2 软证书私钥文件。
 *  流程：GET /auth/challenge 取挑战 → 读取私钥文件 → 交本机后端代签并验签
 *  （前端不碰密码学；单机本地软件可接受的简化，私钥不落盘/不落日志）。
 *  首次启动（accounts 为空）展示引导窗口：创建第一个三员账号。
 */
import { ref } from "vue";
import { useRouter } from "vue-router";

import { ApiRequestError, bootstrap } from "../services/api";
import { useAuthStore } from "../stores/auth";

const router = useRouter();
const auth = useAuthStore();

const usernameInput = ref("");
const privateKeyText = ref("");
const keyFileName = ref("");
const busy = ref(false);
const error = ref("");

const showBootstrap = ref(false);
const bootUsername = ref("");
const bootRole = ref<"sysadmin" | "secadmin" | "auditor">("sysadmin");
const bootPrivateKey = ref(""); // 引导签发的软证书私钥（一次性展示，提示保存）

async function onPickKeyFile(e: Event): Promise<void> {
  const input = e.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) return;
  keyFileName.value = file.name;
  privateKeyText.value = (await file.text()).trim();
}

function onLoggedOut(): void {
  router.push("/journey");
}

async function submitLogin(): Promise<void> {
  error.value = "";
  if (!usernameInput.value.trim()) {
    error.value = "请输入账号名";
    return;
  }
  if (!privateKeyText.value) {
    error.value = "请选择私钥证书文件（管理员签发的 .key/.pem 文本）";
    return;
  }
  busy.value = true;
  try {
    await auth.login(usernameInput.value.trim(), privateKeyText.value);
    onLoggedOut();
  } catch (e) {
    error.value = e instanceof ApiRequestError ? e.message : "登录失败，请重试";
  } finally {
    busy.value = false;
  }
}

async function submitBootstrap(): Promise<void> {
  error.value = "";
  if (!bootUsername.value.trim()) {
    error.value = "请输入引导账号名";
    return;
  }
  busy.value = true;
  try {
    const out = await bootstrap({ username: bootUsername.value.trim(), role: bootRole.value });
    bootPrivateKey.value = out.private_key ?? "";
    // 保持引导面板展开：私钥展示区就在面板里，收起等于诱导用户丢失唯一一次
    // 展示的私钥（后端口径：仅本次展示、引导窗口创建后永久关闭）。用户确认
    // 保存后可手动收起。
    usernameInput.value = out.username;
  } catch (e) {
    error.value = e instanceof ApiRequestError ? e.message : "引导失败";
  } finally {
    busy.value = false;
  }
}

function copyBootKey(): void {
  void navigator.clipboard?.writeText(bootPrivateKey.value);
}
</script>

<template>
  <div class="login-wrap">
    <form
      class="login-card"
      @submit.prevent="submitLogin"
    >
      <h1 class="title">射线评片智能检测系统</h1>
      <p class="subtitle">三员身份认证（SM2 挑战-响应）</p>

      <label class="field">
        <span>账号名</span>
        <input
          v-model="usernameInput"
          type="text"
          autocomplete="username"
          placeholder="系统管理员 / 保密管理员 / 审计员 账号"
        >
      </label>

      <label class="field">
        <span>私钥证书文件</span>
        <input
          type="file"
          accept=".key,.pem,.txt,.json"
          @change="onPickKeyFile"
        >
      </label>
      <p
        v-if="keyFileName"
        class="hint"
      >已选择：{{ keyFileName }}</p>

      <button
        class="primary"
        type="submit"
        :disabled="busy"
      >
        {{ busy ? "登录中…" : "登 录" }}
      </button>

      <p
        v-if="error"
        class="error"
        role="alert"
      >{{ error }}</p>

      <details
        class="boot"
        :open="showBootstrap"
        @toggle="showBootstrap = ($event.target as HTMLDetailsElement).open"
      >
        <summary>首次启动？创建第一个账号（引导窗口）</summary>
        <template v-if="!bootPrivateKey">
          <label class="field">
            <span>引导账号名</span>
            <input
              v-model="bootUsername"
              type="text"
            >
          </label>
          <label class="field">
            <span>角色（一人一岗）</span>
            <select v-model="bootRole">
              <option value="sysadmin">系统管理员</option>
              <option value="secadmin">安全保密管理员</option>
              <option value="auditor">安全审计员</option>
            </select>
          </label>
          <button
            type="button"
            :disabled="busy"
            @click="submitBootstrap"
          >创建账号并签发软证书</button>
        </template>
        <template v-else>
          <p class="warn">
            请立即保存私钥（仅本次展示，系统不留存）：
            <button
              type="button"
              @click="copyBootKey"
            >复制私钥</button>
          </p>
          <textarea
            class="keyout"
            readonly
            :value="bootPrivateKey"
            rows="3"
          />
        </template>
      </details>
    </form>
  </div>
</template>

<style scoped>
.login-wrap {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  background: var(--bg, #eef1f5);
}
.login-card {
  width: 360px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 28px;
  background: #fff;
  border: 1px solid var(--line, #ddd);
  border-radius: 6px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
}
.title { font-size: 18px; margin: 0; text-align: center; }
.subtitle { font-size: 12px; color: #667; margin: 0 0 8px; text-align: center; }
.field { display: flex; flex-direction: column; gap: 4px; font-size: 12px; }
.field input, .field select { padding: 6px 8px; border: 1px solid var(--line, #ccc); border-radius: 3px; }
.primary { padding: 8px; font-weight: 600; cursor: pointer; }
.error { color: #b3261e; font-size: 12px; margin: 0; }
.hint { font-size: 11px; color: #567; margin: 0; }
.boot { font-size: 12px; }
.boot summary { cursor: pointer; color: #456; }
.warn { color: #7a5900; font-size: 12px; }
.keyout { width: 100%; font-size: 10px; word-break: break-all; }
</style>
