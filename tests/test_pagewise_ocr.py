#!/usr/bin/env python3
"""Test script for page-wise OCR功能"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from pdf2epub.ocr_pages import ocr_full_book_pagewise
from pdf2epub.utils.common import load_config
from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession

def main():
    # 配置
    book_title = "The Cambridge History of Japanese Literature"
    pdf_path = Path(f"output/{book_title}/input.pdf")
    output_dir = Path(f"output/{book_title}")

    # 只测试前 5 页
    start_page = 1
    end_page = 5

    print(f"Testing page-wise OCR on {book_title}")
    print(f"Pages: {start_page}-{end_page}")
    print(f"PDF: {pdf_path}")
    print(f"Output: {output_dir}/pages/")
    print()

    # 加载配置
    config = load_config("config.yaml")

    # 设置认证
    ocr_backend = config.get("ocr_backend", "vertex")

    if ocr_backend == "vertex":
        # Vertex AI 认证
        service_account_path = config.get("service_account_path", "sa-keys.json")
        credentials = service_account.Credentials.from_service_account_file(
            service_account_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        session = AuthorizedSession(credentials)
        project_id = config.get("project_id")
        location = config.get("location", "us-central1")

        # 调用逐页 OCR
        ocr_full_book_pagewise(
            pdf_path=pdf_path,
            output_dir=output_dir,
            session=session,
            project_id=project_id,
            location=location,
            start_page=start_page,
            end_page=end_page,
            backend=ocr_backend,
            resume=True,  # 支持断点续传
            config=config  # 传递配置（包含 retry 设置）
        )

    elif ocr_backend == "mistral":
        # Mistral API 认证
        api_key = config.get("mistral_api_key")
        base_url = config.get("mistral_base_url", "https://api.mistral.ai/v1")

        # 调用逐页 OCR
        ocr_full_book_pagewise(
            pdf_path=pdf_path,
            output_dir=output_dir,
            start_page=start_page,
            end_page=end_page,
            backend=ocr_backend,
            api_key=api_key,
            base_url=base_url,
            resume=True,
            config=config  # 传递配置（包含 retry 设置）
        )

    else:
        print(f"Unsupported backend: {ocr_backend}")
        sys.exit(1)

    print("\n✓ Test completed!")
    print(f"Check output at: {output_dir}/pages/")

if __name__ == "__main__":
    main()
