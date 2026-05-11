import streamlit as st
import easyocr
import pandas as pd
import re
import os
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import io
import zipfile
import numpy as np
from pdf2image import convert_from_bytes
import cv2

# -------------------------- 页面配置 --------------------------
st.set_page_config(page_title="发票识别重命名工具", layout="wide")
st.title("📋 发票批量识别、重命名与汇总表格生成")
st.markdown(
    "上传发票图片或 PDF，自动提取关键信息，按规范重命名并导出表格。"
    "\n\n💡 **提示**：请尽量上传清晰的图片，避免反光、倾斜和阴影；"
    "对手机拍摄的发票，可开启「图像增强」提升识别率。"
)

# -------------------------- 缓存 EasyOCR Reader --------------------------
@st.cache_resource
def get_reader():
    return easyocr.Reader(['ch_sim', 'en'], gpu=False)

# -------------------------- 图像增强预处理 --------------------------
def preprocess_image(image_pil):
    img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )
    return Image.fromarray(processed)

# -------------------------- 发票信息提取 --------------------------
def extract_invoice_info(text):
    info = {
        "发票号码": "",
        "开票日期": "",
        "金额": "",
        "销售方": ""
    }
    # 发票号码
    match = re.search(r'(?:发票号码|No)[：: ]*([\d]+)', text, re.IGNORECASE)
    if match:
        info["发票号码"] = match.group(1)
    # 开票日期
    match = re.search(r'(\d{4}[年\-]\d{1,2}[月\-]\d{1,2}[日]?)', text)
    if match:
        date_str = match.group(1)
        date_str = date_str.replace('年', '').replace('月', '').replace('日', '')
        date_str = date_str.replace('-', '')
        if len(date_str) == 8:
            info["开票日期"] = date_str
    # 金额
    match = re.search(r'(?:价税合计|合计金额|小写)[^\d]*[¥￥]?([\d,]+\.\d{2})', text)
    if not match:
        match = re.search(r'[¥￥]\s*([\d,]+\.\d{2})', text)
    if match:
        info["金额"] = match.group(1).replace(',', '')
    # 销售方
    match = re.search(r'名\s*称[：:]\s*([^\n]+)', text)
    if match:
        name = match.group(1).strip()
        info["销售方"] = re.sub(r'\s+', '', name)
    return info

# -------------------------- 确定归档子文件夹名称 --------------------------
def get_archive_folder(info, archive_mode, custom_field=None):
    """
    根据归档模式返回文件夹名称字符串。
    archive_mode: "不归档" / "按月份" / "按销售方" / "自定义字段"
    """
    if archive_mode == "不归档":
        return ""  # 直接放在根目录
    elif archive_mode == "按月份":
        date = info.get("开票日期", "")
        if date and len(date) >= 6:
            year_month = date[:6]  # 如 202301
            return year_month
        else:
            return "未知月份"
    elif archive_mode == "按销售方":
        seller = info.get("销售方", "")
        return seller if seller else "未知销售方"
    elif archive_mode == "自定义字段" and custom_field:
        value = info.get(custom_field, "")
        return value if value else f"未知{custom_field}"
    else:
        return ""

# -------------------------- 文件 OCR 处理 --------------------------
def ocr_file(file_bytes, filename, use_preprocess):
    reader = get_reader()
    suffix = Path(filename).suffix.lower()
    ocr_text = ""
    if suffix in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
        image = Image.open(io.BytesIO(file_bytes))
        if use_preprocess:
            image = preprocess_image(image)
        image_np = np.array(image)
        results = reader.readtext(image_np, detail=0)
        ocr_text = " ".join(results)
    elif suffix == '.pdf':
        try:
            images = convert_from_bytes(file_bytes)
        except Exception as e:
            st.error(f"PDF 转换失败: {e}。请确保服务器已安装 poppler。")
            return "", {}
        all_text = []
        for img in images:
            if use_preprocess:
                img = preprocess_image(img)
            img_np = np.array(img)
            results = reader.readtext(img_np, detail=0)
            all_text.extend(results)
        ocr_text = " ".join(all_text)
    else:
        st.warning(f"暂不支持的文件类型: {suffix}")
        return "", {}
    info = extract_invoice_info(ocr_text)
    return ocr_text, info

# -------------------------- 安全文件名/文件夹名 --------------------------
def safe_name(s):
    s = re.sub(r'[\\/*?:"<>|]', '', s)  # 移除非法字符
    s = s.strip().rstrip('.')  # 末尾不能是点
    return s if s else "未知"

def generate_new_name(info, original_ext):
    number = info.get("发票号码") or "未知"
    date = info.get("开票日期") or "未知"
    amount = info.get("金额") or "未知"
    seller = info.get("销售方") or "未知"
    base = f"{number}_{date}_{amount}_{seller}"
    base = safe_name(base)
    return f"{base}{original_ext}"

# -------------------------- 主界面 --------------------------
def main():
    # ---------- 左侧边栏：归档设置 ----------
    with st.sidebar:
        st.header("⚙️ 归档设置")
        archive_mode = st.selectbox(
            "选择归档方式",
            ["不归档", "按月份", "按销售方", "自定义字段"],
            index=0,
            help="重命名后的文件将按所选维度放入不同的文件夹中。"
        )
        custom_field = None
        if archive_mode == "自定义字段":
            field_options = ["发票号码", "开票日期", "金额", "销售方"]
            custom_field = st.selectbox(
                "选择用于归档的字段",
                field_options,
                help="将根据该字段的值创建文件夹（如有缺失则归入‘未知’文件夹）。"
            )
        use_preprocess = st.checkbox(
            "🛠️ 启用图像增强",
            value=True,
            help="开启后会对图片进行二值化处理，可能提升手机拍照/扫描件的识别率。"
        )

    # ---------- 主区域 ----------
    uploaded_files = st.file_uploader(
        "📤 选择发票文件（可多选，支持 JPG/PNG/PDF/BMP/TIFF）",
        type=["png", "jpg", "jpeg", "pdf", "bmp", "tiff", "tif"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("🚀 开始识别并处理", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            records = []
            temp_dir = tempfile.mkdtemp()
            total = len(uploaded_files)

            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"正在处理: {uploaded_file.name} ({idx+1}/{total})")

                file_bytes = uploaded_file.read()
                original_name = uploaded_file.name
                original_ext = Path(original_name).suffix

                # OCR + 提取信息
                ocr_text, info = ocr_file(file_bytes, original_name, use_preprocess)

                # 生成新文件名
                new_name = generate_new_name(info, original_ext)

                # 确定归档文件夹
                subfolder = get_archive_folder(info, archive_mode, custom_field)
                subfolder = safe_name(subfolder) if subfolder else ""

                # 在临时目录创建对应的子文件夹（如果有）
                dest_folder = os.path.join(temp_dir, subfolder) if subfolder else temp_dir
                if subfolder and not os.path.exists(dest_folder):
                    os.makedirs(dest_folder, exist_ok=True)
                new_path = os.path.join(dest_folder, new_name)
                with open(new_path, "wb") as f:
                    f.write(file_bytes)

                # 记录到汇总表
                records.append({
                    "原文件名": original_name,
                    "新文件名": new_name,
                    "归档文件夹": subfolder if subfolder else "根目录",
                    "发票号码": info["发票号码"],
                    "开票日期": info["开票日期"],
                    "金额（元）": info["金额"],
                    "销售方": info["销售方"],
                    "OCR文本片段": ocr_text[:150]
                })

                progress_bar.progress((idx + 1) / total)

            status_text.text("✅ 处理完成！")
            st.success(f"共处理 {total} 个文件，结果如下：")

            # ---------- 汇总表 ----------
            st.subheader("📊 提取结果汇总")
            df = pd.DataFrame(records)
            st.dataframe(df, use_container_width=True)

            # ---------- 下载 ----------
            col1, col2 = st.columns(2)
            with col1:
                csv = df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 下载汇总表格 (CSV)",
                    data=csv,
                    file_name="发票汇总.csv",
                    mime="text/csv"
                )
            with col2:
                # 打包成 ZIP，保留文件夹结构
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for root, _, files in os.walk(temp_dir):
                        for file in files:
                            full_path = os.path.join(root, file)
                            # arcname 是 ZIP 内的相对路径
                            arcname = os.path.relpath(full_path, temp_dir)
                            zf.write(full_path, arcname)
                zip_buffer.seek(0)
                st.download_button(
                    label="📦 下载全部文件 (ZIP)",
                    data=zip_buffer,
                    file_name="归档发票.zip",
                    mime="application/zip"
                )

            # 清理临时目录
            shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()