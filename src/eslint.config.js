// ESLint 扁平配置（ESLint 9，与后端 ruff 同级的 JS/TS 代码门禁）。
// 说明：本配置为「接线」而非「强制门禁」——`yarn lint` 可选用；
// 全量启用前建议先跑一次 `yarn lint` 处理存量告警（主要为风格类）。
import tseslint from "typescript-eslint";
import vue from "eslint-plugin-vue";

export default [
  {
    ignores: [
      "node_modules",
      "dist",
      "src-tauri/target",
      "**/*.d.ts",
      "src/types/generated.ts", // openapi-typescript 自动生成，不lint
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
      // 前端大量使用 any 桥接后端契约，暂不强约束
      "@typescript-eslint/no-explicit-any": "off",
      "@typescript-eslint/no-unused-vars": ["warn", { argsIgnorePattern: "^_" }],
      "vue/multi-word-component-names": "off",
      "vue/no-v-html": "off",
    },
  },
];
