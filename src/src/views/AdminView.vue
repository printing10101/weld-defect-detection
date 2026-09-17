<script setup lang="ts">
/**
 * 账号管理（仅系统管理员）：三员账号列表 / 启停 / 创建账号 / 重新签发 SM2 密钥。
 * 私钥丢失此后在界面内有自救路径（此前私钥一旦遗失，账号永久无法登录且无任何入口补救）。
 * 私钥为一次性展示（后端不留副本），复制失败必须显式告知，不允许静默。
 */
import { computed, onMounted, ref } from "vue";
import ConfirmDialog from "../components/ConfirmDialog.vue";
import {
  createAccount,
  issueKeypair,
  listAccounts,
  setAccountStatus,
} from "../services/api";
import { useAuthStore } from "../stores/auth";
import { toErrorMessage } from "../utils/errorMessage";
import { copyText } from "../utils/clipboard";
import type { AccountOut } from "../types/api";

const auth = useAuthStore();
const allowed = computed(() => auth.role === "sysadmin");

const accounts = ref<AccountOut[]>([]);
const loading = ref(false);
const err = ref<string | null>(null);
const notice = ref<string | null>(null);

const ROLE_NAMES: Record<string, string> = {
  sysadmin: "系统管理员",
  secadmin: "安全保密管理员",
  auditor: "安全审计员",
};

async function load(): Promise<void> {
  loading.value = true;
  err.value = null;
  try {
    accounts.value = await listAccounts();
  } catch (e) {
    err.value = toErrorMessage(e);
  } finally {
    loading.value = false;
  }
}

onMounted(() => {
  if (allowed.value) void load();
});

/* ── 创建账号：创建成功后立即签发密钥并进入一次性私钥展示 ── */
const newUsername = ref("");
const newRole = ref<"sysadmin" | "secadmin" | "auditor">("auditor");
const creating = ref(false);

const keyPanel = ref<{ username: string; privateKey: string } | null>(null);
const copyState = ref<"" | "ok" | "fail">("");

async function onCreate(): Promise<void> {
  notice.value = null;
  const name = newUsername.value.trim();
  if (!name) {
    err.value = "请输入新账号名。";
    return;
  }
  creating.value = true;
  err.value = null;
  try {
    const acc = await createAccount({ username: name, role: newRole.value });
    const kp = await issueKeypair(acc.account_id);
    keyPanel.value = { username: acc.username, privateKey: kp.private_key };
    copyState.value = "";
    newUsername.value = "";
    await load();
  } catch (e) {
    err.value = toErrorMessage(e);
  } finally {
    creating.value = false;
  }
}

async function onReissue(acc: AccountOut): Promise<void> {
  notice.value = null;
  err.value = null;
  try {
    const kp = await issueKeypair(acc.account_id);
    keyPanel.value = { username: acc.username, privateKey: kp.private_key };
    copyState.value = "";
  } catch (e) {
    err.value = toErrorMessage(e);
  }
}

async function onCopyKey(): Promise<void> {
  if (!keyPanel.value) return;
  copyState.value = (await copyText(keyPanel.value.privateKey)) ? "ok" : "fail";
}

/* ── 启用/停用（停用即时吊销会话，二次确认） ── */
const disableTarget = ref<AccountOut | null>(null);
const toggling = ref(false);

function askToggle(acc: AccountOut): void {
  if (acc.status === "disabled") {
    void doToggle(acc);
  } else {
    disableTarget.value = acc;
  }
}

async function doToggle(acc: AccountOut): Promise<void> {
  toggling.value = true;
  err.value = null;
  try {
    const next = acc.status === "disabled" ? "active" : "disabled";
    await setAccountStatus(acc.account_id, next);
    notice.value =
      next === "disabled" ? `账号「${acc.username}」已停用并吊销其全部会话。` : `账号「${acc.username}」已启用。`;
    await load();
  } catch (e) {
    err.value = toErrorMessage(e);
  } finally {
    toggling.value = false;
    disableTarget.value = null;
  }
}
</script>

<template>
  <div>
    <h1
      class="title-zine"
      data-t="账号管理"
    >
      账号管理
    </h1>
    <div class="lede">
      三员账号全生命周期管理（一人一岗）；账号停用即时吊销会话，私钥可重新签发（一次性展示）
    </div>

    <div
      v-if="!allowed"
      class="empty"
    >
      本页面仅系统管理员可用。当前登录身份无权限查看账号管理。
    </div>

    <template v-else>
      <p
        v-if="err"
        class="err show"
      >
        注意：{{ err }}
      </p>
      <p
        v-else-if="notice"
        class="ok-msg"
      >
        ✓ {{ notice }}
      </p>

      <div class="section-h">
        <span class="no">1</span>新建账号
      </div>
      <div class="create-row">
        <input
          v-model="newUsername"
          placeholder="新账号名（如 zhangsan）"
          @keyup.enter="onCreate"
        >
        <select v-model="newRole">
          <option value="sysadmin">
            系统管理员
          </option>
          <option value="secadmin">
            安全保密管理员
          </option>
          <option value="auditor">
            安全审计员
          </option>
        </select>
        <button
          class="btn"
          type="button"
          :disabled="creating"
          @click="onCreate"
        >
          {{ creating ? "创建中…" : "创建并签发密钥 →" }}
        </button>
      </div>

      <!-- 一次性私钥展示 -->
      <div
        v-if="keyPanel"
        class="key-panel"
        role="alert"
      >
        <p class="kp-title">
          注意：账号「{{ keyPanel.username }}」的 SM2 私钥（仅此一次展示，系统不留存副本）
        </p>
        <textarea
          class="keyout"
          readonly
          :value="keyPanel.privateKey"
          rows="4"
          @focus="($event.target as HTMLTextAreaElement).select()"
        />
        <div class="kp-acts">
          <button
            type="button"
            class="btn"
            @click="onCopyKey"
          >
            复制私钥
          </button>
          <span
            v-if="copyState === 'ok'"
            class="cp-ok"
          >✓ 已复制到剪贴板</span>
          <span
            v-else-if="copyState === 'fail'"
            class="cp-fail"
          >复制失败：请在上面的文本框内手动全选复制，关闭本面板后将无法再次查看。</span>
          <button
            type="button"
            class="btn ghost"
            style="margin-left: auto"
            @click="keyPanel = null"
          >
            我已妥善保存，关闭
          </button>
        </div>
      </div>

      <div class="section-h">
        <span class="no">2</span>账号列表
      </div>
      <p
        v-if="loading"
        class="hint"
      >
        加载中…
      </p>
      <p
        v-else-if="accounts.length === 0"
        class="empty"
      >
        暂无账号。
      </p>
      <table v-else>
        <thead>
          <tr>
            <th>账号名</th>
            <th>岗位角色</th>
            <th>状态</th>
            <th>连续失败次数</th>
            <th>锁定至</th>
            <th>创建时间</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="a in accounts"
            :key="a.account_id"
          >
            <td>{{ a.username }}</td>
            <td>{{ ROLE_NAMES[a.role] ?? a.role }}</td>
            <td>
              <span
                class="lv"
                :class="{ disabled: a.status !== 'active' }"
              >{{ a.status === "active" ? "启用" : "停用" }}</span>
            </td>
            <td>{{ a.failed_attempts }}</td>
            <td>{{ a.locked_until ?? "—" }}</td>
            <td>{{ a.created_at ?? "—" }}</td>
            <td class="ops">
              <button
                type="button"
                class="btn link"
                :disabled="toggling"
                @click="askToggle(a)"
              >
                {{ a.status === "active" ? "停用" : "启用" }}
              </button>
              <button
                type="button"
                class="btn link"
                :disabled="toggling"
                @click="onReissue(a)"
              >
                重新签发密钥…
              </button>
            </td>
          </tr>
        </tbody>
      </table>
      <p class="hint">
        「重新签发密钥」会使旧私钥立即失效（私钥丢失或疑似泄露时使用）；新私钥仅展示一次。
      </p>
    </template>

    <ConfirmDialog
      :open="disableTarget !== null"
      title="停用账号确认"
      :message="`停用后账号「${disableTarget?.username ?? ''}」将立即无法登录，其全部会话被吊销；可随时再启用。确认停用？`"
      confirm-text="停用账号"
      danger
      @confirm="disableTarget && doToggle(disableTarget)"
      @cancel="disableTarget = null"
    />
  </div>
</template>

<style scoped>
.create-row {
  display: flex;
  gap: 10px;
  align-items: center;
  margin-bottom: 16px;
  flex-wrap: wrap;
}
.create-row input {
  width: 240px;
  padding: 6px 9px;
  font-size: 13px;
}
.create-row select {
  padding: 6px 8px;
  font-size: 13px;
}
.key-panel {
  border: 1px solid rgba(214, 134, 21, 0.55);
  background: rgba(214, 134, 21, 0.06);
  border-radius: 8px;
  padding: 12px 14px;
  margin-bottom: 16px;
}
.kp-title {
  margin: 0 0 8px;
  font-size: 13px;
  font-weight: 600;
  color: #7a5900;
}
.keyout {
  width: 100%;
  box-sizing: border-box;
  font-size: 11px;
  word-break: break-all;
}
.kp-acts {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 8px;
  flex-wrap: wrap;
}
.cp-ok {
  color: #1e7a3d;
  font-size: 12px;
}
.cp-fail {
  color: #b03030;
  font-size: 12px;
}
.lv.disabled {
  color: #6a7b99;
}
.ops {
  white-space: nowrap;
}
.ok-msg {
  padding: 8px 12px;
  border-radius: 8px;
  background: rgba(42, 143, 74, 0.12);
  color: #1e7a3d;
  font-size: 13px;
  margin-bottom: 12px;
}
</style>
