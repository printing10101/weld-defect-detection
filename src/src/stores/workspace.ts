/**
 * 工作台全局状态（Pinia）——跨菜单栏/工具栏/各工作区共享。
 *：把原本散落在组件局部的作用域状态（操作员姓名）提升为应用级 store，
 * 使「谁在操作」在各视图/请求间保持一致，并可被单元测试直接验证。
 * 持久化仍落地到 localStorage（单机无用户系统），由 operator 服务封装。
 */
import { defineStore } from "pinia";
import { ref } from "vue";
import { getOperatorName, setOperatorName } from "../services/operator";

export const useWorkspaceStore = defineStore("workspace", () => {
  /** 当前操作员（报告签名与审计留痕）；空回退 "local"。 */
  const operator = ref(getOperatorName());

  function setOperator(name: string): void {
    setOperatorName(name);
    operator.value = getOperatorName();
  }

  return { operator, setOperator };
});