import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/ 
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      // 方案一：直接代理 /rag 路径（改动最小）
      // '/rag': {
      //   target: 'http://localhost:8000',
      //   changeOrigin: true,
      //   secure: false,
      // }
      
      // 方案二：使用 /api 前缀（更规范，如需此方案请注释掉上面的 '/rag' 配置）
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '') // 去掉 /api 前缀再转发到后端
      }
    }
  }
})