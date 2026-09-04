// ESLint 扁平配置（ESLint 9，与后端 ruff 同级的 JS/TS 代码门禁）。
// 说明：本配置为「接线」而非「强制门禁」——`pnpm lint` 可选用；
// 全量启用前建议先跑一次 `pnpm lint` 处理存量告警（主要为风格类）。
import tseslint from "typescript-eslint";
import vue from "eslint-plugin-vue";

export default [
  {
    ignores: [
      "node_modules",
      "dist",
      "src-tauri/target",
      "**/*.d.ts",
    ],
  },
  ...tseslint.configs.recommended,
  ...vue.configs["flat/recommended"],
  {
    files: ["**/*.vue"],
    languageOptions: {
      parserOptions: { parser: tseslint.parser },
    },
  },
  {
    rules: {
      // 实测全库 any 为 0（后端契约经 types/api.ts 镜像），恢复强约束
      "@typescript-eslint/no-explicit-any": "error",
      "@typescript-eslint/no-unused-vars": ["warn", { argsIgnorePattern: "^_" }],
      "vue/multi-word-component-names": "off",
      "vue/no-v-html": "off",
    },
  },
];
