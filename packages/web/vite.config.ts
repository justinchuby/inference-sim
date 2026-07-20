import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig(({ mode }) => ({
  base: mode === "github-pages" ? "/inference-sim/" : "/",
  plugins: [
    react(),
    tailwindcss(),
    {
      name: "disable-cloudflare-rocket-loader",
      transformIndexHtml: {
        order: "post",
        handler(html) {
          return html.replace(
            '<script type="module" crossorigin',
            '<script data-cfasync="false" type="module" crossorigin',
          );
        },
      },
    },
  ],
  server: {
    port: 4173,
  },
  build: {
    sourcemap: true,
  },
}));
