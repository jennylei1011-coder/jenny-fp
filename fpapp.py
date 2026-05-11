import io
import os
import re
import shutil
import tempfile
import unicodedata
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import easyocr
import numpy as np
import pandas as pd
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image, ImageOps


st.set_page_config(page_title="JENNY 发票识别", page_icon="📄", layout="wide")

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".pdf"}
UPLOAD_TYPES = ["png", "jpg", "jpeg", "pdf", "bmp", "tiff", "tif", "zip"]
PARTY_FIELDS = ["名称", "纳税人识别号", "地址电话", "开户行及账号"]
ITEM_FIELDS = ["项目名称", "规格型号", "单位", "数量", "单价", "金额", "税率", "税额"]
INVOICE_FIELDS = [
    "发票号码",
    "开票日期",
    "购买方名称",
    "购买方纳税人识别号",
    "销售方名称",
    "销售方纳税人识别号",
    "价税合计",
]
RESULT_REQUIRED_FIELDS = ["发票号码", "开票日期", "销售方名称", "价税合计"]
RESULT_COLUMNS = [
    "发票号码",
    "开票日期",
    "购买方名称",
    "购买方纳税人识别号",
    "购买方地址电话",
    "购买方开户行及账号",
    "销售方名称",
    "销售方纳税人识别号",
    "销售方地址电话",
    "销售方开户行及账号",
    "项目名称",
    "规格型号",
    "单位",
    "数量",
    "单价",
    "金额",
    "税率",
    "税额",
    "价税合计",
]
MAX_ZIP_FILES = 300
MAX_ZIP_TOTAL_SIZE = 800 * 1024 * 1024
OCR_PROFILES = {
    "快速": {
        "max_width": 900,
        "pdf_dpi": 150,
        "max_pdf_pages": 1,
        "canvas_size": 1280,
        "mag_ratio": 0.9,
    },
    "均衡": {
        "max_width": 1200,
        "pdf_dpi": 180,
        "max_pdf_pages": 3,
        "canvas_size": 1800,
        "mag_ratio": 1.0,
    },
    "高精度": {
        "max_width": 1600,
        "pdf_dpi": 220,
        "max_pdf_pages": 8,
        "canvas_size": 2560,
        "mag_ratio": 1.1,
    },
}


def apply_custom_styles():
    st.markdown(
        """
        <style>
        :root {
            --app-bg: #f7f8f5;
            --panel-bg: #ffffff;
            --text-main: #1f2923;
            --text-muted: #607066;
            --line: #d9e0d8;
            --accent: #2f6f5e;
            --accent-strong: #245647;
        }

        .stApp {
            background: var(--app-bg);
            color: var(--text-main);
        }

        [data-testid="stAppViewContainer"] > .main .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 2.5rem;
        }

        [data-testid="stSidebar"] {
            background: #fbfcfa;
            border-right: 1px solid var(--line);
        }

        .app-hero {
            border-bottom: 1px solid var(--line);
            margin-bottom: 1.25rem;
            padding-bottom: 1rem;
        }

        .app-kicker {
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0;
            margin-bottom: 0.25rem;
            text-transform: uppercase;
        }

        .app-title {
            color: var(--text-main);
            font-size: clamp(2rem, 3.2vw, 3.25rem);
            font-weight: 760;
            letter-spacing: 0;
            line-height: 1.05;
            margin: 0;
        }

        .app-subtitle {
            color: var(--text-muted);
            font-size: 1rem;
            line-height: 1.7;
            max-width: 720px;
            margin-top: 0.75rem;
        }

        .metric-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 1rem 0 1.25rem;
        }

        .metric-item {
            background: var(--panel-bg);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.8rem 0.9rem;
            min-height: 74px;
        }

        .metric-label {
            color: var(--text-muted);
            font-size: 0.78rem;
            margin-bottom: 0.25rem;
        }

        .metric-value {
            color: var(--text-main);
            font-size: 1.45rem;
            font-weight: 740;
            line-height: 1.1;
            word-break: break-word;
        }

        .stButton > button,
        .stDownloadButton > button {
            border: 1px solid var(--accent);
            border-radius: 8px;
            background: var(--accent);
            color: #ffffff;
            font-weight: 700;
            min-height: 2.75rem;
            box-shadow: none;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: var(--accent-strong);
            background: var(--accent-strong);
            color: #ffffff;
        }

        [data-testid="stFileUploader"] section {
            background: var(--panel-bg);
            border: 1px dashed #9aaba1;
            border-radius: 8px;
            padding: 0.85rem;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
        }

        div[data-testid="stAlert"] {
            border-radius: 8px;
            border: 1px solid var(--line);
        }

        h1, h2, h3 {
            letter-spacing: 0;
        }

        @media (max-width: 760px) {
            [data-testid="stAppViewContainer"] > .main .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .metric-strip {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="正在加载本地识别模型...")
def get_reader():
    model_dir = Path(__file__).resolve().parent / "model"
    return easyocr.Reader(
        ["ch_sim", "en"],
        gpu=False,
        model_storage_directory=str(model_dir),
        download_enabled=False,
    )


def preprocess_image(image_pil):
    image_pil = ImageOps.exif_transpose(image_pil).convert("RGB")
    img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    )
    return Image.fromarray(processed)


def resize_image(image_np, max_width=1400):
    h, w = image_np.shape[:2]
    if w > max_width:
        ratio = max_width / w
        image_np = cv2.resize(
            image_np,
            (max_width, max(1, int(h * ratio))),
            interpolation=cv2.INTER_AREA,
        )
    return image_np


def normalize_ocr_text(results):
    lines = [str(item).strip() for item in results if str(item).strip()]
    return "\n".join(lines)


def clean_extracted_value(value):
    value = unicodedata.normalize("NFKC", value or "")
    value = re.sub(r"[ \t\r\f\v]+", "", value)
    value = re.sub(r"[|｜]+", "", value)
    return value.strip(" :：,，.;；")


def normalize_item_line(value):
    value = unicodedata.normalize("NFKC", value or "")
    value = re.sub(r"[\x00-\x1f\x7f-\x9f\u200b-\u200f\u2028-\u202f]", " ", value)
    value = re.sub(r"[|｜]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" :：,，.;；")


def parse_invoice_date(text):
    patterns = [
        r"(\d{4})\s*[年\-/\.]\s*(\d{1,2})\s*[月\-/\.]\s*(\d{1,2})\s*日?",
        r"(\d{4})(\d{2})(\d{2})",
    ]
    for pattern in patterns:
        for year, month, day in re.findall(pattern, text):
            if not 1990 <= int(year) <= 2100:
                continue
            try:
                parsed = datetime(int(year), int(month), int(day))
            except ValueError:
                continue
            return parsed.strftime("%Y%m%d")
    return ""


def parse_invoice_number(text):
    compact = clean_extracted_value(text)
    patterns = [
        r"(?:发票号码|票据号码|No\.?|NO\.?)[:：]?\s*([A-Z0-9]{6,24})",
        r"(?:号码)[:：]?\s*([0-9]{8,24})",
    ]
    for pattern in patterns:
        match = re.search(pattern, compact, re.IGNORECASE)
        if match:
            return re.sub(r"\D", "", match.group(1)) or match.group(1)
    return ""


def parse_amount(text):
    compact = clean_extracted_value(text)
    amount_pattern = r"([0-9]{1,3}(?:,[0-9]{3})*\.[0-9]{2}|[0-9]+\.[0-9]{2})"
    keyword_patterns = [
        rf"(?:价税合计|小写|合计金额|税价合计|总金额)[^0-9¥￥]{{0,12}}[¥￥]?{amount_pattern}",
        rf"[¥￥]\s*{amount_pattern}",
    ]
    for pattern in keyword_patterns:
        matches = re.findall(pattern, compact)
        if matches:
            value = matches[-1]
            if isinstance(value, tuple):
                value = value[-1]
            return value.replace(",", "")
    return ""


def clean_name(value, max_len=60):
    if not value:
        return "未知"
    value = unicodedata.normalize("NFKC", str(value))
    value = re.sub(r"[\x00-\x1f\x7f-\x9f\u200b-\u200f\u2028-\u202f]", "", value)
    value = re.sub(r'[\\/*?:"<>|]', "_", value)
    value = re.sub(r"\s+", "", value)
    value = value.strip().rstrip(".")
    if not value:
        return "未知"
    return value[:max_len]


def trim_company_name(value):
    value = clean_extracted_value(value)
    value = re.split(
        r"(?:纳税人识别号|统一社会信用代码|地址|电话|开户行|账号|购买方|销售方|密码区|备注)",
        value,
        maxsplit=1,
    )[0]
    match = re.search(
        r"(.{2,60}?(?:有限责任公司|股份有限公司|有限公司|公司|事务所|集团|中心|厂|店))",
        value,
    )
    if match:
        value = match.group(1)
    return clean_name(value, 60) if value else ""


def parse_party_section(text, party_keywords, stop_keywords):
    compact_lines = [clean_extracted_value(line) for line in text.splitlines() if line.strip()]
    compact_text = "\n".join(compact_lines)
    start_pattern = "|".join(party_keywords)
    stop_pattern = "|".join(stop_keywords)
    match = re.search(
        rf"(?:{start_pattern})(.*?)(?=(?:{stop_pattern})|$)",
        compact_text,
        re.DOTALL,
    )
    return match.group(1) if match else ""


def parse_labeled_value(section, label_patterns, max_len=120):
    section_lines = [line for line in section.splitlines() if line.strip()]
    for label in label_patterns:
        match = re.search(rf"(?:{label})[:：]?([^\n]{{2,{max_len}}})", section)
        if match:
            return clean_extracted_value(match.group(1))
        for index, line in enumerate(section_lines):
            if re.search(label, line):
                value = clean_extracted_value(re.sub(label, "", line, count=1))
                if len(value) >= 2:
                    return value[:max_len]
                if index + 1 < len(section_lines):
                    return clean_extracted_value(section_lines[index + 1])[:max_len]
    return ""


def parse_party_info(text, role):
    if role == "购买方":
        section = parse_party_section(
            text,
            ["购买方", "购货方", "付款方"],
            ["销售方", "销货方", "收款方", "项目名称", "货物或应税劳务"],
        )
    else:
        section = parse_party_section(
            text,
            ["销售方", "销货方", "收款方", "销售单位"],
            ["备注", "收款人", "复核", "开票人", "价税合计"],
        )

    name = parse_labeled_value(section, [r"名称", r"名\s*称"], 90)
    if not name:
        company_match = re.search(
            r"([^\n]{2,90}?(?:有限责任公司|股份有限公司|有限公司|公司|事务所|集团|中心|厂|店))",
            section,
        )
        if company_match:
            name = company_match.group(1)

    tax_id = parse_labeled_value(
        section,
        [r"纳税人识别号", r"统一社会信用代码", r"税号", r"识别号"],
        80,
    )
    if not tax_id:
        tax_match = re.search(r"\b([0-9A-Z]{15,20})\b", clean_extracted_value(section))
        tax_id = tax_match.group(1) if tax_match else ""

    address_phone = parse_labeled_value(section, [r"地址、电话", r"地址电话", r"地址"], 120)
    bank_account = parse_labeled_value(
        section,
        [r"开户行及账号", r"开户行及帐号", r"开户银行及账号", r"开户行账号"],
        120,
    )

    return {
        f"{role}名称": trim_company_name(name),
        f"{role}纳税人识别号": clean_name(tax_id, 30) if tax_id else "",
        f"{role}地址电话": address_phone,
        f"{role}开户行及账号": bank_account,
    }


def parse_seller(text):
    return parse_party_info(text, "销售方").get("销售方名称", "")


def normalize_money(value):
    if not value:
        return ""
    value = value.replace(",", "").replace("￥", "").replace("¥", "")
    match = re.search(r"-?\d+(?:\.\d{1,2})?", value)
    return match.group(0) if match else ""


def parse_invoice_items(text):
    lines = [normalize_item_line(line) for line in text.splitlines() if line.strip()]
    item_lines = []
    started = False

    for line in lines:
        if re.search(r"(项目名称|货物或应税劳务|服务名称)", line):
            started = True
            continue
        if started and re.search(r"(合计|价税合计|销售方|备注|收款人|复核|开票人)", line):
            break
        if started:
            item_lines.append(line)

    if not item_lines:
        item_lines = [
            line
            for line in lines
            if re.search(r"\d+\.\d{2}", line)
            and not re.search(r"(价税合计|合计金额|小写|税额合计)", line)
        ]

    items = []
    for line in item_lines:
        if not line or len(line) < 3:
            continue

        tax_rate_match = re.search(r"(\d{1,2}%|免税|不征税|普通征税)", line)
        tax_rate = tax_rate_match.group(1) if tax_rate_match else ""
        money_values = [normalize_money(item) for item in re.findall(r"-?\d+(?:,\d{3})*\.\d{2}", line)]
        money_values = [item for item in money_values if item]

        if len(money_values) < 2 and not tax_rate:
            continue

        tax_amount = money_values[-1] if money_values else ""
        amount = money_values[-2] if len(money_values) >= 2 else ""
        unit_price = money_values[-3] if len(money_values) >= 3 else ""

        prefix = line
        for value in money_values:
            prefix = prefix.replace(value, " ")
        if tax_rate:
            prefix = prefix.replace(tax_rate, " ")
        prefix = re.sub(r"\s+", " ", prefix).strip()

        parts = [part for part in re.split(r"\s+|,|，", prefix) if part]
        project_name = parts[0] if parts else prefix
        spec_model = parts[1] if len(parts) >= 4 else ""
        unit = parts[2] if len(parts) >= 4 else (parts[1] if len(parts) >= 3 else "")
        quantity = parts[3] if len(parts) >= 4 else ""

        items.append(
            {
                "项目名称": project_name,
                "规格型号": spec_model,
                "单位": unit,
                "数量": quantity,
                "单价": unit_price,
                "金额": amount,
                "税率": tax_rate,
                "税额": tax_amount,
            }
        )

    if items:
        return items

    raw_tokens = [line for line in lines if line]
    start_index = 0
    for index, token in enumerate(raw_tokens):
        if re.search(r"(项目名称|货物或应税劳务|服务名称)", token):
            start_index = index + 1
            break

    end_index = len(raw_tokens)
    for index, token in enumerate(raw_tokens[start_index:], start=start_index):
        if re.search(r"(合计|价税合计|销售方|备注|收款人|复核|开票人)", token):
            end_index = index
            break
    tokens = [
        token
        for token in raw_tokens[start_index:end_index]
        if not re.fullmatch(
            r"(项目名称|规格型号|单位|数量|单价|金额|税率|税额|货物或应税劳务|服务名称)",
            token,
        )
    ]

    def is_money_token(value):
        return bool(re.fullmatch(r"-?\d+(?:,\d{3})*\.\d{2}", value))

    def is_number_token(value):
        return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", value))

    parsed_items = []
    tax_indexes = [
        index
        for index, token in enumerate(tokens)
        if re.fullmatch(r"\d{1,2}%|免税|不征税|普通征税", token)
    ]

    segment_start = 0
    for tax_index in tax_indexes:
        before = tokens[segment_start:tax_index]
        after = tokens[tax_index + 1 :]
        money_before = [item for item in before if is_money_token(item)]
        money_after = [item for item in after if is_money_token(item)]
        if not money_before and not money_after:
            continue

        tax_amount = normalize_money(money_after[0]) if money_after else ""
        amount = normalize_money(money_before[-1]) if money_before else ""
        unit_price = normalize_money(money_before[-2]) if len(money_before) >= 2 else ""

        amount_index = before.index(money_before[-1]) if money_before else len(before)
        prefix = before[:amount_index]
        numeric_prefix = [item for item in prefix if is_number_token(item) and not is_money_token(item)]
        quantity = numeric_prefix[-1] if numeric_prefix else ""
        text_prefix = [item for item in prefix if not is_number_token(item) and not is_money_token(item)]

        project_name = text_prefix[0] if text_prefix else ""
        spec_model = text_prefix[1] if len(text_prefix) >= 3 else ""
        unit = text_prefix[-1] if len(text_prefix) >= 2 else ""

        parsed_items.append(
            {
                "项目名称": project_name,
                "规格型号": spec_model,
                "单位": unit,
                "数量": quantity,
                "单价": unit_price,
                "金额": amount,
                "税率": tokens[tax_index],
                "税额": tax_amount,
            }
        )
        if money_after:
            segment_start = tax_index + 1 + after.index(money_after[0]) + 1

    if parsed_items:
        return parsed_items
    return [{field: "" for field in ITEM_FIELDS}]


def extract_invoice_info(text):
    buyer = parse_party_info(text, "购买方")
    seller = parse_party_info(text, "销售方")
    return {
        "发票号码": parse_invoice_number(text),
        "开票日期": parse_invoice_date(text),
        "价税合计": parse_amount(text),
        **buyer,
        **seller,
        "项目明细": parse_invoice_items(text),
    }


def generate_new_name(info, original_ext):
    number = clean_name(info.get("发票号码") or "未知", 24)
    date = clean_name(info.get("开票日期") or "未知", 8)
    amount = clean_name(info.get("价税合计") or "未知", 14)
    seller = clean_name(info.get("销售方名称") or "未知", 48)
    base = clean_name(f"{number}_{date}_{amount}_{seller}", 180)
    return f"{base}{original_ext.lower()}"


def get_archive_folder(info, archive_mode):
    if archive_mode == "不归档":
        return ""
    if archive_mode == "按月份":
        date = info.get("开票日期", "")
        return date[:6] if len(date) >= 6 else "未知月份"
    if archive_mode == "按销售方":
        return clean_name(info.get("销售方名称", ""), 50)
    return ""


def make_unique_filename(folder, filename):
    folder_path = Path(folder)
    candidate = folder_path / filename
    if not candidate.exists():
        return filename

    stem = candidate.stem
    suffix = candidate.suffix
    for index in range(2, 1000):
        next_name = f"{stem}_{index}{suffix}"
        if not (folder_path / next_name).exists():
            return next_name
    return f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"


def read_image_text(reader, image, use_preprocess, profile):
    image = ImageOps.exif_transpose(image)
    if use_preprocess:
        image = preprocess_image(image)
    else:
        image = image.convert("RGB")
    image_np = resize_image(np.array(image), max_width=profile["max_width"])
    results = reader.readtext(
        image_np,
        detail=0,
        paragraph=False,
        decoder="greedy",
        canvas_size=profile["canvas_size"],
        mag_ratio=profile["mag_ratio"],
    )
    return normalize_ocr_text(results)


def ocr_file(file_bytes, filename, use_preprocess, profile):
    reader = get_reader()
    suffix = Path(filename).suffix.lower()

    if suffix in SUPPORTED_EXTENSIONS - {".pdf"}:
        try:
            image = Image.open(io.BytesIO(file_bytes))
            ocr_text = read_image_text(reader, image, use_preprocess, profile)
        except Exception as exc:
            raise RuntimeError(f"图片读取或识别失败：{exc}") from exc
    elif suffix == ".pdf":
        try:
            images = convert_from_bytes(
                file_bytes,
                dpi=profile["pdf_dpi"],
                first_page=1,
                last_page=profile["max_pdf_pages"],
            )
        except Exception as exc:
            raise RuntimeError(f"PDF 转换失败，请确认已安装 poppler：{exc}") from exc

        all_text = []
        for image in images:
            all_text.append(read_image_text(reader, image, use_preprocess, profile))
        ocr_text = "\n".join(item for item in all_text if item)
    else:
        raise RuntimeError("不支持的文件格式")

    return ocr_text, extract_invoice_info(ocr_text)


def extract_files_from_zip(zip_bytes):
    files = []
    total_size = 0
    skipped = 0

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue

                suffix = Path(member.filename).suffix.lower()
                if suffix not in SUPPORTED_EXTENSIONS:
                    skipped += 1
                    continue

                total_size += member.file_size
                if len(files) >= MAX_ZIP_FILES or total_size > MAX_ZIP_TOTAL_SIZE:
                    skipped += 1
                    continue

                files.append((zf.read(member), clean_name(Path(member.filename).name, 160)))
    except zipfile.BadZipFile as exc:
        raise RuntimeError("ZIP 文件无法读取或已经损坏") from exc

    return files, skipped


def render_header():
    st.markdown(
        """
        <div class="app-hero">
            <div class="app-kicker">Invoice OCR Workspace</div>
            <h1 class="app-title">JENNY 发票识别</h1>
            <div class="app-subtitle">
                批量识别图片、PDF 或 ZIP 中的发票，提取购销方、项目明细、税额和价税合计，并生成可下载的规范表格。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(records):
    df = pd.DataFrame(records)
    total_rows = len(df)
    invoice_count = int(df["源文件名"].nunique()) if total_rows else 0
    missing = int((df[RESULT_REQUIRED_FIELDS] == "").any(axis=1).sum()) if total_rows else 0
    failed = (
        int(df["处理备注"].astype(str).str.contains("失败|错误", regex=True).sum())
        if total_rows
        else 0
    )
    amount_df = df.drop_duplicates(subset=["源文件名", "发票号码", "价税合计"])
    amount_sum = pd.to_numeric(amount_df["价税合计"], errors="coerce").sum() if total_rows else 0

    st.markdown(
        f"""
        <div class="metric-strip">
            <div class="metric-item"><div class="metric-label">处理文件</div><div class="metric-value">{invoice_count}</div></div>
            <div class="metric-item"><div class="metric-label">明细行数</div><div class="metric-value">{total_rows}</div></div>
            <div class="metric-item"><div class="metric-label">待核对</div><div class="metric-value">{missing + failed}</div></div>
            <div class="metric-item"><div class="metric-label">价税合计</div><div class="metric-value">{amount_sum:,.2f}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_excel_bytes(df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="发票汇总")
        worksheet = writer.sheets["发票汇总"]
        for column_cells in worksheet.columns:
            max_length = max(len(str(cell.value or "")) for cell in column_cells)
            worksheet.column_dimensions[column_cells[0].column_letter].width = min(
                max(max_length + 2, 12),
                42,
            )
    buffer.seek(0)
    return buffer.getvalue()


def reset_previous_results():
    old_dir = st.session_state.get("temp_dir")
    if old_dir:
        shutil.rmtree(old_dir, ignore_errors=True)
    st.session_state.pop("records", None)
    st.session_state.pop("temp_dir", None)
    st.session_state.pop("results_ready", None)


def process_uploads(uploaded_files, archive_mode, use_preprocess, ocr_profile_name):
    reset_previous_results()
    profile = OCR_PROFILES[ocr_profile_name]

    to_process = []
    zip_messages = []

    for uploaded_file in uploaded_files:
        name = uploaded_file.name
        data = uploaded_file.read()
        if name.lower().endswith(".zip"):
            extracted, skipped = extract_files_from_zip(data)
            zip_messages.append((name, len(extracted), skipped))
            to_process.extend(extracted)
        else:
            to_process.append((data, clean_name(name, 160)))

    if not to_process:
        raise RuntimeError("没有可处理的发票文件")

    temp_dir = tempfile.mkdtemp(prefix="jenny_invoice_")
    records = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(to_process)

    for index, (file_bytes, original_name) in enumerate(to_process, start=1):
        status_text.text(f"正在处理 {index}/{total}：{original_name}")
        original_ext = Path(original_name).suffix.lower()
        note = ""
        ocr_text = ""

        try:
            ocr_text, info = ocr_file(file_bytes, original_name, use_preprocess, profile)
        except Exception as exc:
            info = {field: "" for field in INVOICE_FIELDS}
            info["项目明细"] = [{field: "" for field in ITEM_FIELDS}]
            note = str(exc)

        new_name = generate_new_name(info, original_ext)
        subfolder = get_archive_folder(info, archive_mode)
        subfolder_clean = "" if not subfolder else clean_name(subfolder, 80)
        dest_folder = Path(temp_dir) / subfolder_clean if subfolder_clean else Path(temp_dir)
        dest_folder.mkdir(parents=True, exist_ok=True)

        try:
            saved_name = make_unique_filename(dest_folder, new_name)
            (dest_folder / saved_name).write_bytes(file_bytes)
        except Exception as exc:
            saved_name = f"invoice_{uuid.uuid4().hex[:8]}{original_ext}"
            try:
                (dest_folder / saved_name).write_bytes(file_bytes)
                note = f"{note}；原文件名保存失败，已使用备用名称：{exc}".strip("；")
            except Exception as save_exc:
                saved_name = "保存失败"
                note = f"{note}；保存失败：{save_exc}".strip("；")

        missing_fields = [field for field in INVOICE_FIELDS if not info.get(field)]
        if missing_fields:
            note = f"{note}；待核对：{','.join(missing_fields)}".strip("；")

        for item in info.get("项目明细") or [{field: "" for field in ITEM_FIELDS}]:
            records.append(
                {
                    "源文件名": original_name,
                    "发票号码": info.get("发票号码", ""),
                    "开票日期": info.get("开票日期", ""),
                    "购买方名称": info.get("购买方名称", ""),
                    "购买方纳税人识别号": info.get("购买方纳税人识别号", ""),
                    "购买方地址电话": info.get("购买方地址电话", ""),
                    "购买方开户行及账号": info.get("购买方开户行及账号", ""),
                    "销售方名称": info.get("销售方名称", ""),
                    "销售方纳税人识别号": info.get("销售方纳税人识别号", ""),
                    "销售方地址电话": info.get("销售方地址电话", ""),
                    "销售方开户行及账号": info.get("销售方开户行及账号", ""),
                    "项目名称": item.get("项目名称", ""),
                    "规格型号": item.get("规格型号", ""),
                    "单位": item.get("单位", ""),
                    "数量": item.get("数量", ""),
                    "单价": item.get("单价", ""),
                    "金额": item.get("金额", ""),
                    "税率": item.get("税率", ""),
                    "税额": item.get("税额", ""),
                    "价税合计": info.get("价税合计", ""),
                    "处理备注": note,
                }
            )
        progress_bar.progress(index / total)

    status_text.text("处理完成")
    return records, temp_dir, zip_messages


def render_downloads(df, temp_dir):
    col1, col2, col3 = st.columns(3)
    with col1:
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "下载 CSV",
            data=csv,
            file_name="JENNY_发票汇总.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "下载 Excel",
            data=build_excel_bytes(df),
            file_name="JENNY_发票汇总.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with col3:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(temp_dir):
                for filename in files:
                    full_path = Path(root) / filename
                    zf.write(full_path, full_path.relative_to(temp_dir))
        st.download_button(
            "下载归档 ZIP",
            data=zip_buffer.getvalue(),
            file_name="JENNY_归档发票.zip",
            mime="application/zip",
            use_container_width=True,
        )


def main():
    apply_custom_styles()
    render_header()

    with st.sidebar:
        st.header("处理设置")
        ocr_profile_name = st.selectbox(
            "识别模式",
            list(OCR_PROFILES.keys()),
            index=0,
            help="快速模式适合线上批量处理；高精度会更慢，适合少量模糊发票。",
        )
        archive_mode = st.selectbox(
            "归档方式",
            ["不归档", "按月份", "按销售方"],
            help="识别后的文件会按所选维度放入不同文件夹。",
        )
        use_preprocess = st.checkbox(
            "图像增强",
            value=False,
            help="适合手机拍摄、灰底或轻微模糊的发票；清晰扫描件可关闭以加快处理。",
        )

    uploaded_files = st.file_uploader(
        "选择发票文件",
        type=UPLOAD_TYPES,
        accept_multiple_files=True,
        help="支持图片、PDF，以及包含这些文件的 ZIP。",
    )

    if uploaded_files:
        st.caption(f"已选择 {len(uploaded_files)} 个上传项")

    start = st.button(
        "开始识别",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True,
    )

    if start:
        try:
            records, temp_dir, zip_messages = process_uploads(
                uploaded_files,
                archive_mode,
                use_preprocess,
                ocr_profile_name,
            )
        except Exception as exc:
            st.error(str(exc))
        else:
            st.session_state["records"] = records
            st.session_state["temp_dir"] = temp_dir
            st.session_state["results_ready"] = True

            for filename, count, skipped in zip_messages:
                if skipped:
                    st.warning(f"{filename} 已提取 {count} 个可处理文件，跳过 {skipped} 个文件。")
                else:
                    st.info(f"{filename} 已提取 {count} 个可处理文件。")

    if st.session_state.get("results_ready"):
        records = st.session_state["records"]
        df = pd.DataFrame(records)
        df = df.reindex(columns=RESULT_COLUMNS, fill_value="")
        render_metrics(records)

        st.subheader("识别结果")
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "购买方地址电话": st.column_config.TextColumn(width="large"),
                "购买方开户行及账号": st.column_config.TextColumn(width="large"),
                "销售方地址电话": st.column_config.TextColumn(width="large"),
                "销售方开户行及账号": st.column_config.TextColumn(width="large"),
            },
        )
        render_downloads(df, st.session_state["temp_dir"])


if __name__ == "__main__":
    main()
