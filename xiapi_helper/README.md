# Shopee Seller CLI (Playwright) Auto-Selector & Recorder

用 **Playwright (Python)** 完成：
- 命令列交互：選擇店鋪或直接輸入 URL（例如 `https://seller.shopee.tw/portal/...`）
- 注入店鋪 Cookie（支持 **JSON 陣列** 或 **cookie_string**）
- 自動勾選 **未顯示「廣告進行中」** 的商品；若本頁無可勾選則自動翻頁，直到達標
- 全程記錄行為與頁面（事件 JSONL、每頁 HTML/PNG、匯總 CSV）

## 安裝
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
python -m playwright install
```

## 配置
1) 從瀏覽器導出 Cookie JSON 陣列（示例見 `cookies.sample.json`）。
2) 編輯 `stores.json`（多店鋪隨意添加）：
```json
{
  "default_headless": false,
  "default_target_to_pick": 30,
  "stores": [
    { "name": "店鋪A", "cookie_file": "cookies_storeA.json" },
    { "name": "店鋪B", "cookie_file": "cookies_storeB.json" }
  ]
}
```

> 也可使用 `cookie_string` 模式（"name=value; name2=value2; ..."）

## 使用
```bash
python cli_playwright_ad_recorder.py
```
- 選擇店鋪（序號或名稱）
- 輸入要打開的 URL（回車=只打開首頁 `https://seller.shopee.tw/`）
- 設定目標勾選數 / 是否 headless
- 開始執行；輸出在 `runs/<店鋪名_時間戳>/`

## 調整
- 主要選擇器在腳本頂部 `SEL`：
  - 勾選：`label.eds-checkbox` / `input.eds-checkbox__input`
  - 廣告標記：`a.existing-ads-text`（或行內文字包含）
  - 下一頁：`button.eds-pager__button-next`
- 如需並發多店鋪、HAR、或把事件轉成「回放腳本」，可以擴展。

