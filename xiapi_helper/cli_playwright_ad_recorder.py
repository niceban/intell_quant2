# -*- coding: utf-8 -*-
import os, sys, time, json, csv, random, traceback
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

try:
    import pandas as pd
except Exception:
    pd = None

SEL = {
    "table_body": ".eds-table__body",
    "rows": ".eds-table__body tbody > tr, .eds-table__body tr",
    "checkbox_label": "label.eds-checkbox",
    "checkbox_input": "input.eds-checkbox__input",
    "ads_badge": "a.existing-ads-text",
    "next_btn": "button.eds-pager__button-next",
    "row_name": ".ellipsis-content, .product-name, .table-col-name-box, .eds-table__cell"
}

def rand_sleep(page, a=120, b=350):
    page.wait_for_timeout(random.randint(a, b))

def page_signature(page):
    rows = page.locator(SEL["rows"])
    count = rows.count()
    if count == 0:
        return "#0"
    def name_at(i):
        try:
            cell = rows.nth(i).locator(SEL["row_name"]).first
            return cell.inner_text(timeout=500).strip()
        except Exception:
            try:
                return rows.nth(i).inner_text(timeout=500).strip()[:32]
            except Exception:
                return ""
    first = name_at(0)
    last = name_at(count-1)
    return f"{first}|{last}#{count}"

def wait_for_table(page, timeout=15000):
    page.wait_for_selector(SEL["table_body"], timeout=timeout)
    page.wait_for_selector(SEL["rows"], timeout=timeout)

def is_row_ad(page, row):
    if row.locator(SEL["ads_badge"]).count():
        return True
    try:
        t = row.inner_text(timeout=500)
        if "廣告進行中" in t or "广告进行中" in t:
            return True
    except Exception:
        pass
    return False

def is_row_checked(row):
    try:
        inp = row.locator(SEL["checkbox_input"])
        if inp.count():
            return inp.is_checked()
    except Exception:
        pass
    return False

def click_row_checkbox(page, row):
    label = row.locator(SEL["checkbox_label"]).first
    if label.count():
        label.scroll_into_view_if_needed()
        label.click()
        return True
    box = row.locator(SEL["checkbox_input"]).first
    if box.count():
        box.scroll_into_view_if_needed()
        box.click()
        return True
    return False

def click_next_and_wait(page, timeout_ms=8000):
    btn = page.locator(SEL["next_btn"])
    if not btn.count():
        return False
    aria = (btn.get_attribute("aria-disabled") or "").lower()
    if aria == "true":
        return False
    before = page_signature(page)
    btn.click()
    start = time.time()
    while (time.time() - start) * 1000 < timeout_ms:
        now = page_signature(page)
        if now != before:
            page.wait_for_timeout(400)
            return True
        page.wait_for_timeout(150)
    return False

def save_snapshot(page, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{prefix}.html"), "w", encoding="utf-8") as f:
        f.write(page.content())
    page.screenshot(path=os.path.join(out_dir, f"{prefix}.png"), full_page=True)

@dataclass
class Store:
    name: str
    cookie_file: Optional[str] = None
    cookie_string: Optional[str] = None

def load_config(path="stores.json"):
    with open(path, "r", encoding="utf-8") as f:
        conf = json.load(f)
    defaults = {
        "default_headless": conf.get("default_headless", False),
        "default_target_to_pick": conf.get("default_target_to_pick", 30),
    }
    stores = []
    for s in conf.get("stores", []):
        stores.append(Store(
            name=s["name"],
            cookie_file=s.get("cookie_file"),
            cookie_string=s.get("cookie_string")
        ))
    return defaults, stores

def choose_store(stores: List[Store]) -> Store:
    print("可用店鋪：")
    for i, s in enumerate(stores):
        src = "cookie_file" if s.cookie_file else "cookie_string"
        print(f"  [{i}] {s.name}  ({src})")
    raw = input("請輸入店鋪序號或名稱（回車=0）：").strip()
    if raw == "":
        return stores[0]
    if raw.isdigit():
        return stores[int(raw)]
    for s in stores:
        if s.name == raw:
            return s
    raise SystemExit("未找到對應的店鋪")

def inject_cookies(context, base_url, store: Store):
    # 要先打到同域，才能設置 cookie
    page = context.new_page()
    page.goto(base_url, wait_until="domcontentloaded")
    host = urlparse(base_url).hostname or "seller.shopee.tw"

    if store.cookie_file and os.path.exists(store.cookie_file):
        arr = json.load(open(store.cookie_file, "r", encoding="utf-8"))
        cookies = []
        for c in arr:
            cookies.append({
                "name": c["name"],
                "value": c["value"],
                "domain": c.get("domain", "." + host),
                "path": c.get("path", "/"),
                "httpOnly": bool(c.get("httpOnly", False)),
                "secure": bool(c.get("secure", True))
            })
        context.add_cookies(cookies)
        print("[*] 已注入 cookie（文件）")
    elif store.cookie_string:
        cookies = []
        for part in store.cookie_string.split(";"):
            part = part.strip()
            if not part or "=" not in part: continue
            name, value = part.split("=", 1)
            cookies.append({
                "name": name.strip(),
                "value": value.strip(),
                "domain": "." + host,
                "path": "/",
                "httpOnly": False,
                "secure": True
            })
        context.add_cookies(cookies)
        print("[*] 已注入 cookie（字串）")
    else:
        print("[!] 未提供 cookie，可能需要手動登入")

    page.close()

def run_flow(context, base_url, target_url, target_to_pick, out_dir):
    page = context.new_page()
    page.goto(target_url or base_url)
    print("[*] 打開：", target_url or base_url)

    # 嘗試等待表格（如果只是首頁，可能沒有）
    has_table = True
    try:
        wait_for_table(page, timeout=15000)
    except Exception:
        has_table = False

    picked_total, page_counter = 0, 0
    save_snapshot(page, out_dir, f"page_{page_counter:03d}_init")

    if not has_table:
        print("[!] 未檢測到列表表格（可能在首頁或頁面不同）。流程結束。")
        page.close()
        return

    while picked_total < target_to_pick:
        rows = page.locator(SEL["rows"])
        count = rows.count()
        if count == 0:
            print("[!] 列表為空，停止。")
            break

        picked_this_page = 0
        for i in range(count):
            if picked_total >= target_to_pick:
                break
            row = rows.nth(i)
            if is_row_ad(page, row): continue
            if is_row_checked(row): continue
            if click_row_checkbox(page, row):
                picked_total += 1
                picked_this_page += 1
                print(f"  ✅ 勾選 {picked_total}/{target_to_pick}")
                rand_sleep(page)

        save_snapshot(page, out_dir, f"page_{page_counter:03d}_after_select")
        if picked_total >= target_to_pick: break

        ok = click_next_and_wait(page)
        if not ok:
            print("[!] 無法翻頁或到最後一頁，停止。")
            break
        page_counter += 1
        save_snapshot(page, out_dir, f"page_{page_counter:03d}_init_after_next")

    page.close()

def main():
    print("=== Shopee Seller CLI (Playwright) Auto-Selector & Recorder ===")
    base_host = "seller.shopee.tw"
    base_url = f"https://{base_host}/"

    defaults, stores = load_config("stores.json")
    store = choose_store(stores)

    custom_url = input("請輸入要打開的 URL（回車=首頁）：").strip() or None
    raw_target = input(f"目標勾選數（預設 {defaults['default_target_to_pick']}）：").strip()
    target_to_pick = int(raw_target) if raw_target else int(defaults["default_target_to_pick"])
    raw_headless = input(f"Headless 無頭模式？(y/N，預設 {'Y' if defaults['default_headless'] else 'N'})：").strip().lower()
    headless = defaults["default_headless"]
    if raw_headless in ("y", "yes"): headless = True
    if raw_headless in ("n", "no"): headless = False

    out_dir = os.path.join("runs", f"{store.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context()
        inject_cookies(context, base_url, store)
        run_flow(context, base_url, custom_url, target_to_pick, out_dir)
        browser.close()

    print("\n完成。輸出目錄：", out_dir)

if __name__ == "__main__":
    main()
