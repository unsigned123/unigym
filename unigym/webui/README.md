# webui -- UniGym 门户前端（v0.29.1）

**本目录与 Python 的唯一关系是 HTTP API**（docs/contracts/daemon_api_contract.md）。
纯静态站点：零构建、零框架、零打包器（vanilla JS/CSS，ES modules）；
浏览器只连 daemon 门户单口（8600）。

## 结构

```
index.html          门户（live runs / spawn / recent runs——v0.43 账本卡）
run.html            run 详情页（/view/<run_id> 服务的就是本文件）
assets/app.js       hash 路由 + 共享工具箱（el/fmt/poller/统一错误区）
assets/api.js       fetch 封装 -- 唯一知道端点的文件（方法名=契约条目）
assets/portal.js    index 页视图
assets/runpage.js   run 页视图（status 行 + 组装壳；v0.37.1 拆分）
assets/stageview.js run 页 live-view 面板（MJPEG 自愈 + WebRTC 切换 +
                    相机 stream_bind + 死 run 提示；v0.37.1 拆分）
assets/runpanels.js run 页只读侧卡（metrics 瓦片+sparkline 2s / log tail 3s /
                    behavior report 一次性；v0.37.1 拆分）
assets/consolecard.js run 页 console 面板（eval/step/gate；v0.37.1 拆分）
assets/freecam.js   "free" 相机自由飞行控制（Pointer Lock + WASD/Space/Shift/
                    滚轮；位姿本地积分 + 节流推送 camera op；v0.29.1）
assets/telemetry.js live-view 帧遥测行（1s 轮询 /status）
assets/charts.js    rewards sparkline
assets/style.css    设计系统（CSS 变量暗色主题；全文件无硬编码色值）
```

## 纪律（design §1/§4/§9）

- 视图只调 `api.js`；端点路径不出现在任何其他文件
- 单 JS 文件 ≤300 行、职责单一（run 页按面板拆分为 stageview/runpanels/
  consolecard 独立模块，runpage.js 只留 status 行与组装壳）；无 npm / 无框架
- 时间/数字格式化只在 `app.js` 各恰一份
- 错误统一走页面顶部 `#err` 区（对应 API 的 `{error, hint}`）
- API 变更先改契约，再改 `api.js`，最后才是视图
