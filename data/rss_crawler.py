#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实新闻 RSS 爬虫 (v2)
从多个公开 RSS 源抓取真实新闻，保存为 SemanticIndexer::buildIndex() 兼容的 RSS 2.0 XML 格式

数据源:
  1. 新浪新闻 API (JSON) - 约 100,000 篇
  2. 人民网 RSS (58 频道) - 约 5,800 篇
  3. 中国日报 RSS (9 频道) - 约 1,900 篇
  4. 其他源 (NPR, 36氪, etc.) - 约 150 篇
  目标: 100,000+ 篇真实新闻文章
"""

import os
import sys
import time
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from xml.sax.saxutils import escape

try:
    import requests
except ImportError:
    print("错误: 需要 requests 库。请运行: pip3 install requests")
    sys.exit(1)

# ============================================================
# 配置
# ============================================================

OUTPUT_DIR = "/home/marisa/code1/VectorDB/data/yuliao/xml"
FALLBACK_DIR = "/home/marisa/code1/VectorDB/data/yuliao/xml"
REQUEST_TIMEOUT = 30
ITEMS_PER_FILE = 500  # 与现有最大文件 (dataparse.xml=500) 一致
REQUEST_DELAY = 0.5
MAX_RETRIES = 3
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# ============================================================
# 人民网 RSS 频道列表 (来自 OPML, 58 频道)
# ============================================================

PEOPLE_RSS_CHANNELS = [
    ("politics", "人民网-时政"), ("world", "人民网-国际"), ("finance", "人民网-财经"),
    ("money", "人民网-金融"), ("energy", "人民网-能源"), ("ccnews", "人民网-央企"),
    ("sports", "人民网-体育"), ("legal", "人民网-法制"), ("edu", "人民网-教育"),
    ("culture", "人民网-文化"), ("society", "人民网-社会"), ("media", "人民网-传媒"),
    ("theory", "人民网-理论"), ("ent", "人民网-娱乐"), ("opinion", "人民网-观点"),
    ("auto", "人民网-汽车"), ("haixia", "人民网-海峡两岸"), ("it", "人民网-IT"),
    ("env", "人民网-环保"), ("gongyi", "人民网-公益"), ("caipiao", "人民网-彩票"),
    ("scitech", "人民网-科技"), ("history", "人民网-文史"), ("art", "人民网-收藏"),
    ("book", "人民网-读书"), ("shipin", "人民网-食品"), ("game", "人民网-游戏"),
    ("homea", "人民网-家电"), ("house", "人民网-房产"), ("health", "人民网-健康"),
    ("ip", "人民网-知识产权"), ("cpc", "人民网-共产党新闻网"), ("dangjian", "人民网-党建"),
    ("dangshi", "人民网-党史"), ("npc", "人民网-中国人大新闻"), ("cppcc", "人民网-中国政协新闻"),
    ("military", "人民网-军事"), ("tv", "人民网-电视"), ("unn", "人民网-地方"),
    ("travel", "人民网-旅游"), ("renshi", "人民网-人事"), ("leaders", "人民网-领导"),
    ("pic", "人民网-图片"), ("yuqing", "人民网-舆情"), ("hm", "人民网-港澳"),
    ("tc", "人民网-通信"), ("lady", "人民网-时尚"), ("hongmu", "人民网-红木"),
    ("ru", "人民网-俄罗斯"), ("japan", "人民网-日本"), ("uk", "人民网-英国"),
    ("usa", "人民网-美国"), ("korea", "人民网-韩国"), ("sh", "人民网-上海"),
    ("phb", "人民网-热点新闻"), ("liuyan", "人民网-地方领导留言板"),
    ("chinapic", "人民网-图说中国"),
]

# ============================================================
# 工具函数
# ============================================================

def ensure_output_dir():
    for d in [OUTPUT_DIR, FALLBACK_DIR]:
        try:
            os.makedirs(d, exist_ok=True)
            test_file = os.path.join(d, ".write_test")
            with open(test_file, "w") as f:
                f.write("ok")
            os.remove(test_file)
            return d
        except (OSError, PermissionError):
            continue
    print("错误: 无法写入任何输出目录!")
    sys.exit(1)


def save_xml_file(items, category, source_name, language, generator, output_dir, file_index=0):
    """将 items 保存为 RSS 2.0 XML 文件"""
    if not items:
        return 0

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    first_link = items[0].get("link", "https://example.com") or "https://example.com"

    lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<rss version="2.0">',
        '    <channel>',
        f'        <title><![CDATA[{escape(source_name)}]]></title>',
        '        <image>',
        f'            <title><![CDATA[{escape(source_name)}]]></title>',
        f'            <link>{escape(first_link)}</link>',
        '            <url>https://example.com/logo.gif</url>',
        '        </image>',
        f'        <description><![CDATA[{escape(source_name)} RSS Feed]]></description>',
        f'        <link>{escape(first_link)}</link>',
        f'        <language>{language}</language>',
        f'        <generator>{generator}</generator>',
        f'        <copyright><![CDATA[Copyright {generator}]]></copyright>',
        f'        <pubDate>{now}</pubDate>',
    ]

    for item in items:
        title = escape(item.get("title", "无标题"))
        link = escape(item.get("link", ""))
        pub_date = item.get("pubDate", now)
        author = escape(item.get("author", generator))
        description = item.get("description", f"<p>{title}</p>")
        lines.append('        <item>')
        lines.append(f'            <title><![CDATA[{title}]]></title>')
        lines.append(f'            <link>{link}</link>')
        lines.append(f'            <pubDate>{pub_date}</pubDate>')
        lines.append(f'            <author>{author}</author>')
        lines.append(f'            <description><![CDATA[{description}]]></description>')
        lines.append('        </item>')

    lines.append('    </channel>')
    lines.append('</rss>')

    suffix = f"_{file_index}" if file_index > 0 else ""
    filename = f"{category}{suffix}.xml"
    filepath = os.path.join(output_dir, filename)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"  ✓ 已保存: {filepath} ({len(items)} 篇)")
        return len(items)
    except Exception as e:
        print(f"  ✗ 写入失败 {filepath}: {e}")
        return 0


def save_items(items, category, source_name, language, generator, output_dir):
    """将 items 分批保存为 XML 文件"""
    if not items:
        return 0
    total = 0
    for i in range(0, len(items), ITEMS_PER_FILE):
        chunk = items[i:i + ITEMS_PER_FILE]
        file_index = i // ITEMS_PER_FILE
        total += save_xml_file(chunk, category, source_name, language,
                               generator, output_dir, file_index)
    return total


def parse_date(date_str):
    """解析日期字符串，返回标准化格式"""
    if not date_str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%a, %d %b %Y %H:%M:%S",
        "%d %b %Y %H:%M:%S %z",
        "%Y/%m/%d %H:%M:%S",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# 1. 新浪新闻 API 爬取 (JSON API, 多个分类)
# ============================================================

SINA_CATEGORIES = [
    ("sina_yaowen", "新浪要闻", "2509"),
    ("sina_guonei", "新浪国内", "2510"),
    ("sina_guoji", "新浪国际", "2511"),
    ("sina_shehui", "新浪社会", "2512"),
    ("sina_tiyu", "新浪体育", "2513"),
    ("sina_yule", "新浪娱乐", "2514"),
    ("sina_keji", "新浪科技", "2515"),
    ("sina_gundong", "新浪滚动", "2516"),
    ("sina_caijing", "新浪财经", "2517"),
    ("sina_junshi", "新浪军事", "2518"),
]


def fetch_sina_category(output_dir, category, name, lid):
    """从新浪新闻 API 抓取单个分类的文章"""
    base_url = "https://feed.sina.com.cn/api/roll/get"
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    all_items = []
    empty_pages = 0

    for page in range(1, 2001):  # max 2000 pages
        params = {"pageid": "153", "lid": lid, "num": "50", "page": str(page)}

        for attempt in range(MAX_RETRIES):
            try:
                r = requests.get(base_url, params=params, headers=headers,
                                 timeout=REQUEST_TIMEOUT)
                r.raise_for_status()
                data = r.json()
                result = data.get("result", {})
                if not result:
                    empty_pages += 1
                    break

                total_available = int(result.get("total", 0))
                items_data = result.get("data", [])

                if not items_data:
                    empty_pages += 1
                    if empty_pages >= 3:
                        print(f"    连续 {empty_pages} 页无数据，结束")
                        break
                    break

                empty_pages = 0

                for item in items_data:
                    title = item.get("title", "").strip()
                    url = item.get("url", item.get("wapurl", "")).strip()
                    if not title and not url:
                        continue

                    ctime = item.get("intime") or item.get("ctime", "0")
                    try:
                        ts = int(ctime)
                        if ts > 0:
                            dt = datetime.fromtimestamp(ts, tz=timezone(timedelta(hours=8)))
                            pub_date = dt.strftime("%Y-%m-%d %H:%M:%S")
                        else:
                            pub_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        pub_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    author = item.get("author", item.get("media_name", name)).strip() or name
                    summary = item.get("summary", item.get("intro", "")).strip()
                    description = f"<p>{escape(summary or title)}</p>"

                    all_items.append({
                        "title": title,
                        "link": url,
                        "pubDate": pub_date,
                        "author": author,
                        "description": description,
                    })

                print(f"    [第 {page} 页] +{len(items_data)} (总计: {len(all_items)})", end="\r")
                time.sleep(REQUEST_DELAY)
                break

            except requests.exceptions.RequestException as e:
                print(f"\n  ⚠ 第 {page} 页请求失败: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 * (attempt + 1))
                else:
                    empty_pages += 1
                    break

        if empty_pages >= 3:
            break

    print(f"\n  📊 总计: {len(all_items)} 篇")
    if all_items:
        return save_items(all_items, category, name, "zh-cn", "新浪新闻", output_dir)
    return 0


def fetch_sina_news(output_dir):
    """抓取所有新浪新闻分类"""
    print(f"\n{'='*60}")
    print("📰 新浪新闻 API (JSON) - 10 个分类")
    print(f"{'='*60}")

    total = 0
    for idx, (category, name, lid) in enumerate(SINA_CATEGORIES, 1):
        print(f"\n[{idx}/10] {name} (lid={lid})")
        total += fetch_sina_category(output_dir, category, name, lid)

    print(f"\n📊 新浪新闻总计: {total} 篇")
    return total


# ============================================================
# 2. 人民网 RSS (58 频道)
# ============================================================

def fetch_people_rss(output_dir):
    """抓取所有 58 个人民网 RSS 频道"""
    print(f"\n{'='*60}")
    print("📰 人民网 RSS (58 频道)")
    print(f"{'='*60}")

    headers = {"User-Agent": USER_AGENT, "Accept": "application/rss+xml"}
    total_saved = 0
    success_count = 0

    for idx, (channel, name) in enumerate(PEOPLE_RSS_CHANNELS, 1):
        url = f"http://www.people.com.cn/rss/{channel}.xml"
        print(f"\n[{idx}/58] {name}")
        print(f"   URL: {url}")

        xml_text = None
        for attempt in range(MAX_RETRIES):
            try:
                r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
                r.raise_for_status()
                xml_text = r.text
                break
            except requests.exceptions.RequestException as e:
                print(f"  ⚠ 尝试 {attempt+1}/{MAX_RETRIES}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1)

        if not xml_text:
            print(f"  ✗ 获取失败，跳过")
            continue

        try:
            root = ET.fromstring(xml_text.encode("utf-8"))
        except ET.ParseError as e:
            print(f"  ✗ XML 解析失败: {e}")
            continue

        channel_elem = root.find("channel")
        if channel_elem is None:
            print(f"  ✗ 无法找到 channel 元素")
            continue

        items = []
        for item_elem in channel_elem.findall("item"):
            def get_text(tag):
                el = item_elem.find(tag)
                return el.text.strip() if el is not None and el.text else ""

            title = get_text("title")
            link = get_text("link")
            pub_date = get_text("pubDate")
            author = get_text("author") or name
            description = get_text("description") or f"<p>{escape(title)}</p>"

            if not title and not link:
                continue

            if pub_date:
                pub_date = parse_date(pub_date)
            else:
                pub_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            items.append({
                "title": title,
                "link": link,
                "pubDate": pub_date,
                "author": author,
                "description": description,
            })

        if items:
            saved = save_items(items, f"people_{channel}", name,
                              "zh-cn", "人民网", output_dir)
            total_saved += saved
            success_count += 1
            print(f"  ✓ {len(items)} 篇")
        else:
            print(f"  ⚠ 未提取到文章")

        time.sleep(REQUEST_DELAY)

    print(f"\n📊 人民网总计: {success_count}/58 频道成功, {total_saved} 篇")
    return total_saved


# ============================================================
# 3. 中国日报 RSS (9 频道)
# ============================================================

CHINADAILY_CHANNELS = [
    ("world", "中国日报-国际"), ("china", "中国日报-国内"),
    ("culture", "中国日报-文化"), ("lifestyle", "中国日报-生活"),
    ("sports", "中国日报-体育"), ("opinion", "中国日报-观点"),
    ("photo", "中国日报-图片"), ("video", "中国日报-视频"),
    ("entertainment", "中国日报-娱乐"),
]


def fetch_chinadaily_rss(output_dir):
    """抓取中国日报 RSS 频道"""
    print(f"\n{'='*60}")
    print("📰 中国日报 RSS (9 频道)")
    print(f"{'='*60}")

    headers = {"User-Agent": USER_AGENT}
    total_saved = 0
    success_count = 0

    for idx, (channel, name) in enumerate(CHINADAILY_CHANNELS, 1):
        url = f"https://www.chinadaily.com.cn/rss/{channel}_rss.xml"
        print(f"\n[{idx}/9] {name}")
        print(f"   URL: {url}")

        xml_text = None
        for attempt in range(MAX_RETRIES):
            try:
                r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
                r.raise_for_status()
                xml_text = r.text
                break
            except requests.exceptions.RequestException as e:
                print(f"  ⚠ 尝试 {attempt+1}/{MAX_RETRIES}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1)

        if not xml_text:
            print(f"  ✗ 获取失败，跳过")
            continue

        try:
            root = ET.fromstring(xml_text.encode("utf-8"))
        except ET.ParseError as e:
            print(f"  ✗ XML 解析失败: {e}")
            continue

        channel_elem = root.find("channel")
        if channel_elem is None:
            print(f"  ✗ 无法找到 channel 元素")
            continue

        items = []
        for item_elem in channel_elem.findall("item"):
            def get_text(tag):
                el = item_elem.find(tag)
                return el.text.strip() if el is not None and el.text else ""

            title = get_text("title")
            link = get_text("link")
            pub_date = get_text("pubDate")
            author = get_text("author") or name
            description = get_text("description") or f"<p>{escape(title)}</p>"

            if not title and not link:
                continue

            if pub_date:
                pub_date = parse_date(pub_date)
            else:
                pub_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            items.append({
                "title": title,
                "link": link,
                "pubDate": pub_date,
                "author": author,
                "description": description,
            })

        if items:
            saved = save_items(items, f"chinadaily_{channel}", name,
                              "en", "中国日报", output_dir)
            total_saved += saved
            success_count += 1
            print(f"  ✓ {len(items)} 篇")
        else:
            print(f"  ⚠ 未提取到文章")

        time.sleep(REQUEST_DELAY)

    print(f"\n📊 中国日报总计: {success_count}/9 频道成功, {total_saved} 篇")
    return total_saved


# ============================================================
# 4. 其他源
# ============================================================

OTHER_SOURCES = [
    # (url, category, name, language, generator, is_atom)
    ("https://feeds.npr.org/1001/rss.xml", "npr_news", "NPR News", "en", "NPR", False),
    ("https://feeds.npr.org/1003/rss.xml", "npr_politics", "NPR Politics", "en", "NPR", False),
    ("https://feeds.npr.org/1007/rss.xml", "npr_tech", "NPR Technology", "en", "NPR", False),
    ("https://36kr.com/feed", "kr_36", "36氪", "zh-cn", "36氪", True),
    ("https://sspai.com/feed", "sspai", "少数派", "zh-cn", "少数派", True),
    ("https://www.ifanr.com/feed", "ifanr", "爱范儿", "zh-cn", "爱范儿", True),
    ("https://www.williamlong.info/rss.xml", "williamlong", "月光博客", "zh-cn", "月光博客", False),
    ("https://www.solidot.org/index.rss", "solidot", "Solidot", "zh-cn", "Solidot", False),
    ("http://www.xinhuanet.com/english/rss/worldrss.xml", "xinhua_en", "新华网英文", "en", "新华网", False),
    ("https://rss.sina.com.cn/news/china/focus15.xml", "sina_focus", "新浪新闻聚焦", "zh-cn", "新浪新闻", False),
]


def fetch_other_sources(output_dir):
    """抓取其他 RSS 源"""
    print(f"\n{'='*60}")
    print("📰 其他 RSS 源")
    print(f"{'='*60}")

    headers = {"User-Agent": USER_AGENT}
    total_saved = 0
    success_count = 0

    for idx, (url, category, name, language, generator, is_atom) in enumerate(OTHER_SOURCES, 1):
        print(f"\n[{idx}/{len(OTHER_SOURCES)}] {name}")
        print(f"   URL: {url}")

        xml_text = None
        for attempt in range(MAX_RETRIES):
            try:
                r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
                r.raise_for_status()
                xml_text = r.text
                break
            except requests.exceptions.RequestException as e:
                print(f"  ⚠ 尝试 {attempt+1}/{MAX_RETRIES}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2)

        if not xml_text:
            print(f"  ✗ 获取失败，跳过")
            continue

        try:
            root = ET.fromstring(xml_text.encode("utf-8"))
        except ET.ParseError as e:
            print(f"  ✗ XML 解析失败: {e}")
            continue

        items = []

        if is_atom:
            # Atom 格式: <feed><entry>...
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall(".//atom:entry", ns):
                title_el = entry.find("atom:title", ns)
                title = title_el.text.strip() if title_el is not None and title_el.text else ""

                link_el = entry.find("atom:link", ns)
                link = link_el.get("href", "").strip() if link_el is not None else ""

                if not title and not link:
                    continue

                published = entry.find("atom:published", ns)
                updated = entry.find("atom:updated", ns)
                date_str = ""
                if published is not None:
                    date_str = published.text or ""
                elif updated is not None:
                    date_str = updated.text or ""
                pub_date = parse_date(date_str)

                author_el = entry.find("atom:author/atom:name", ns)
                author = author_el.text.strip() if author_el is not None and author_el.text else name

                content_el = entry.find("atom:content", ns)
                summary_el = entry.find("atom:summary", ns)
                description = ""
                if content_el is not None and content_el.text:
                    description = content_el.text.strip()
                elif summary_el is not None and summary_el.text:
                    description = summary_el.text.strip()
                if not description:
                    description = f"<p>{escape(title)}</p>"

                items.append({
                    "title": title, "link": link, "pubDate": pub_date,
                    "author": author, "description": description,
                })
        else:
            # RSS 2.0: <rss><channel><item>...
            channel_elem = root.find("channel")
            if channel_elem is None:
                for child in root:
                    if child.tag.endswith("channel"):
                        channel_elem = child
                        break
            if channel_elem is None:
                print(f"  ⚠ 无法找到 channel")
                continue

            for item_elem in channel_elem.findall("item"):
                def get_text(tag):
                    el = item_elem.find(tag)
                    return el.text.strip() if el is not None and el.text else ""

                title = get_text("title")
                link = get_text("link")
                pub_date = get_text("pubDate") or get_text("{http://purl.org/dc/elements/1.1/}date")
                author = get_text("author") or get_text("{http://purl.org/dc/elements/1.1/}creator") or name
                description = get_text("content:encoded") or get_text("description") or f"<p>{escape(title)}</p>"

                if not title and not link:
                    continue

                if pub_date:
                    pub_date = parse_date(pub_date)
                else:
                    pub_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                items.append({
                    "title": title, "link": link, "pubDate": pub_date,
                    "author": author, "description": description,
                })

        if items:
            saved = save_items(items, category, name, language, generator, output_dir)
            total_saved += saved
            success_count += 1
            print(f"  ✓ {len(items)} 篇")
        else:
            print(f"  ⚠ 未提取到文章")

        time.sleep(REQUEST_DELAY)

    print(f"\n📊 其他源总计: {success_count}/{len(OTHER_SOURCES)} 成功, {total_saved} 篇")
    return total_saved


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 70)
    print("  真实新闻 RSS 爬虫 v2 - Real News RSS Crawler")
    print(f"  启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("  数据源:")
    print("    1. 新浪新闻 API (JSON, 10分类) - 大幅扩充")
    print("    2. 人民网 RSS (58 频道) - ~5,800 篇")
    print("    3. 中国日报 RSS (9 频道) - ~1,900 篇")
    print("    4. 其他源 (NPR/新华网等) - ~150 篇")
    print("=" * 70)

    output_dir = ensure_output_dir()
    print(f"\n📁 输出目录: {output_dir}")

    grand_total = 0
    start_time = time.time()

    # 1. 新浪新闻 (最大数据源, 10个分类)
    grand_total += fetch_sina_news(output_dir)

    # 2. 人民网
    grand_total += fetch_people_rss(output_dir)

    # 3. 中国日报
    grand_total += fetch_chinadaily_rss(output_dir)

    # 4. 其他源
    grand_total += fetch_other_sources(output_dir)

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  ✅ 爬取完成!")
    print(f"  总文章数: {grand_total}")
    print(f"  输出目录: {output_dir}")
    print(f"  耗时: {elapsed/60:.1f} 分钟")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
