#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DdddOcr FastAPI service.

修复点：
1. /slide_match 使用位置参数调用 DdddOcr.slide_match，不再传 target_bytes/background_bytes/flag。
2. /slide_comparison 使用位置参数调用 DdddOcr.slide_comparison。
3. 保留 flag 字段用于兼容旧客户端请求，但服务端不再向 SDK 传递该参数。
"""

import base64
import binascii
import io
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Union

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

try:
    from pydantic import BaseModel, Field, field_validator
except ImportError:  # pydantic v1 兼容
    from pydantic import BaseModel, Field, validator as field_validator

try:
    from . import DdddOcr, DdddOcrInputError, InvalidImageError, MAX_IMAGE_BYTES as CORE_MAX_IMAGE_BYTES
except ImportError:  # 兼容直接运行场景
    import ddddocr

    DdddOcr = ddddocr.DdddOcr
    DdddOcrInputError = getattr(ddddocr, "DdddOcrInputError", Exception)
    InvalidImageError = getattr(ddddocr, "InvalidImageError", Exception)
    CORE_MAX_IMAGE_BYTES = getattr(ddddocr, "MAX_IMAGE_BYTES", 8 * 1024 * 1024)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ddddocr-api")

ocr_instances: Dict[str, Dict[str, Any]] = {}


def _validate_base64_payload(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} 不能为空")
    try:
        decoded = base64.b64decode(value, validate=True)
    except binascii.Error as exc:
        raise ValueError(f"{field_name} 不是合法的Base64字符串") from exc
    if not decoded:
        raise ValueError(f"{field_name} 内容为空")
    if len(decoded) > CORE_MAX_IMAGE_BYTES:
        raise ValueError(f"{field_name} 大小超过 {CORE_MAX_IMAGE_BYTES // 1024}KB 限制")
    return value


def _decode_base64_bytes(value: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except binascii.Error as exc:
        raise HTTPException(status_code=400, detail="Base64 内容错误") from exc


def _coerce_bool_param(value: Union[bool, str], field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    raise HTTPException(status_code=400, detail=f"{field_name} 只能是布尔值")


def _ensure_colors_list(data: Any) -> List[str]:
    if data in (None, "", "null"):
        return []
    if not isinstance(data, list):
        raise HTTPException(status_code=400, detail="colors 必须是字符串列表")
    normalized: List[str] = []
    for item in data:
        if not isinstance(item, str):
            raise HTTPException(status_code=400, detail="colors 列表中必须是字符串")
        stripped = item.strip()
        if stripped:
            normalized.append(stripped)
    return normalized


def _validate_custom_range_dict(parsed: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
    if not isinstance(parsed, dict):
        raise ValueError("custom_color_ranges 必须是字典")
    for key, ranges in parsed.items():
        if not isinstance(key, str):
            raise ValueError("custom_color_ranges 的键必须为字符串")
        if not isinstance(ranges, list):
            raise ValueError("custom_color_ranges 的值必须为列表")
        for segment in ranges:
            if not isinstance(segment, list) or len(segment) != 3:
                raise ValueError("颜色区间必须是长度为3的列表")
            for value in segment:
                if not isinstance(value, int):
                    raise ValueError("颜色区间中的值需要为整数")
                if not 0 <= value <= 255:
                    raise ValueError("颜色区间的值需位于0-255之间")
    return parsed


def _ensure_custom_ranges(data: Any) -> Optional[Dict[str, List[List[int]]]]:
    if data in (None, "null", ""):
        return None
    if isinstance(data, str):
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="custom_color_ranges JSON 解析失败") from exc
    else:
        parsed = data
    if parsed is None:
        return None
    try:
        return _validate_custom_range_dict(parsed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _normalize_slide_match_result(result: Any) -> Dict[str, Any]:
    if isinstance(result, dict):
        target = result.get("target")
        target_x = result.get("target_x")
        target_y = result.get("target_y")
        if isinstance(target, list) and len(target) >= 2:
            if target_x is None:
                target_x = target[0]
            if target_y is None:
                target_y = target[1]
        if target is None and target_x is not None and target_y is not None:
            target = [int(target_x), int(target_y), int(target_x), int(target_y)]
        if target_x is None or target_y is None or target is None:
            raise HTTPException(status_code=500, detail=f"滑块匹配结果格式异常: {result}")
        return {
            "target_x": int(target_x),
            "target_y": int(target_y),
            "target": [int(x) for x in target],
        }
    if isinstance(result, (list, tuple)) and len(result) >= 2:
        return {
            "target_x": int(result[0]),
            "target_y": int(result[1]),
            "target": [int(x) for x in result[:4]] if len(result) >= 4 else [int(result[0]), int(result[1]), int(result[0]), int(result[1])],
        }
    raise HTTPException(status_code=500, detail=f"滑块匹配结果格式异常: {result}")


class Base64Image(BaseModel):
    image: str = Field(..., description="Base64编码的图片数据")

    @field_validator("image")
    def validate_image(cls, value):
        return _validate_base64_payload(value, "image")


class OCRRequest(Base64Image):
    probability: bool = Field(False, description="是否返回识别概率")
    colors: List[str] = Field(default_factory=list, description="颜色过滤列表")
    custom_color_ranges: Optional[Dict[str, List[List[int]]]] = Field(None, description="自定义颜色范围")

    @field_validator("colors")
    def validate_colors(cls, value):
        return _ensure_colors_list(value)

    @field_validator("custom_color_ranges")
    def validate_custom_ranges(cls, value):
        if value is None:
            return value
        return _validate_custom_range_dict(value)


class SlideMatchRequest(BaseModel):
    target_image: str = Field(..., description="目标图片的Base64编码")
    background_image: str = Field(..., description="背景图片的Base64编码")
    simple_target: bool = Field(False, description="是否使用简化目标")
    flag: bool = Field(False, description="兼容旧客户端字段，当前服务端不再传给SDK")

    @field_validator("target_image")
    def validate_target_image(cls, value):
        return _validate_base64_payload(value, "target_image")

    @field_validator("background_image")
    def validate_background_image(cls, value):
        return _validate_base64_payload(value, "background_image")


class SlideComparisonRequest(BaseModel):
    target_image: str = Field(..., description="目标图片的Base64编码")
    background_image: str = Field(..., description="背景图片的Base64编码")

    @field_validator("target_image")
    def validate_target_image(cls, value):
        return _validate_base64_payload(value, "target_image")

    @field_validator("background_image")
    def validate_background_image(cls, value):
        return _validate_base64_payload(value, "background_image")


class CharsetRangeRequest(BaseModel):
    charset_range: List[str] = Field(..., description="字符范围")

    @field_validator("charset_range")
    def validate_charset(cls, value):
        if not isinstance(value, list):
            raise ValueError("charset_range 需要为字符串列表")
        normalized = []
        for item in value:
            if not isinstance(item, str) or not item:
                raise ValueError("charset_range 需要为非空字符串")
            normalized.append(item)
        return normalized


class OCRResponse(BaseModel):
    result: Union[str, Dict[str, Any]]
    probability: Optional[Any] = None
    processing_time: float


class DetectionResponse(BaseModel):
    result: List[List[int]]
    processing_time: float


class SlideMatchResult(BaseModel):
    target_x: int
    target_y: int
    target: List[int]


class SlideMatchResponse(BaseModel):
    result: SlideMatchResult
    processing_time: float


class SlideComparisonResponse(BaseModel):
    result: Dict[str, Any]
    processing_time: float


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"true", "1", "yes", "y", "on"}


def get_ocr_instance(
    config_key: str,
    ocr: bool = True,
    det: bool = False,
    old: bool = False,
    beta: bool = False,
    use_gpu: bool = False,
    device_id: int = 0,
    show_ad: bool = True,
    import_onnx_path: str = "",
    charsets_path: str = "",
):
    if config_key in ocr_instances:
        ocr_instances[config_key]["last_used"] = time.time()
        return ocr_instances[config_key]["instance"]

    logger.info("创建新的OCR实例，配置: %s", config_key)
    try:
        instance = DdddOcr(
            ocr=ocr,
            det=det,
            old=old,
            beta=beta,
            use_gpu=use_gpu,
            device_id=device_id,
            show_ad=show_ad,
            import_onnx_path=import_onnx_path,
            charsets_path=charsets_path,
        )
        ocr_instances[config_key] = {"instance": instance, "last_used": time.time()}
        return instance
    except (DdddOcrInputError, InvalidImageError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("创建OCR实例失败: %s", exc)
        raise HTTPException(status_code=500, detail=f"初始化OCR失败: {exc}") from exc


def cleanup_inactive_instances(max_idle_time: int = 3600):
    current_time = time.time()
    for key in list(ocr_instances.keys()):
        if current_time - ocr_instances[key]["last_used"] > max_idle_time:
            del ocr_instances[key]
            logger.info("已清理不活跃的OCR实例: %s", key)


app = FastAPI(
    title="DdddOcr API",
    description="DdddOcr通用验证码识别API服务",
    version="1.6.1-fixed",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 默认配置
default_ocr = _env_bool("DDDDOCR_OCR", True)
default_det = _env_bool("DDDDOCR_DET", False)
default_old = _env_bool("DDDDOCR_OLD", False)
default_beta = _env_bool("DDDDOCR_BETA", False)
default_use_gpu = _env_bool("DDDDOCR_USE_GPU", False)
default_device_id = int(os.environ.get("DDDDOCR_DEVICE_ID", "0"))
default_show_ad = _env_bool("DDDDOCR_SHOW_AD", True)
default_import_onnx_path = os.environ.get("DDDDOCR_IMPORT_ONNX_PATH", "")
default_charsets_path = os.environ.get("DDDDOCR_CHARSETS_PATH", "")


@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": time.time()}


@app.post("/ocr", response_model=OCRResponse)
async def ocr_recognition(
    request: OCRRequest,
    background_tasks: BackgroundTasks,
    ocr: bool = Query(default_ocr),
    det: bool = Query(default_det),
    old: bool = Query(default_old),
    beta: bool = Query(default_beta),
    use_gpu: bool = Query(default_use_gpu),
    device_id: int = Query(default_device_id),
    show_ad: bool = Query(default_show_ad),
):
    image = None
    try:
        image = Image.open(io.BytesIO(_decode_base64_bytes(request.image)))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail="无法读取图片") from exc

    config_key = f"ocr={ocr}-det={det}-old={old}-beta={beta}-gpu={use_gpu}-dev={device_id}"
    ocr_instance = get_ocr_instance(config_key, ocr, det, old, beta, use_gpu, device_id, show_ad, default_import_onnx_path, default_charsets_path)
    start_time = time.time()
    try:
        if request.probability:
            result = ocr_instance.classification(image, probability=True, colors=request.colors, custom_color_ranges=request.custom_color_ranges)
            return {"result": result, "probability": result, "processing_time": time.time() - start_time}
        result = ocr_instance.classification(image, colors=request.colors, custom_color_ranges=request.custom_color_ranges)
        return {"result": result, "processing_time": time.time() - start_time}
    except (DdddOcrInputError, InvalidImageError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("OCR识别失败: %s", exc)
        raise HTTPException(status_code=500, detail=f"OCR识别失败: {exc}") from exc
    finally:
        if image is not None:
            try:
                image.close()
            except Exception:
                pass
        background_tasks.add_task(cleanup_inactive_instances)


@app.post("/ocr/file", response_model=OCRResponse)
async def ocr_recognition_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    probability: Union[bool, str] = Form(False),
    colors: str = Form("[]"),
    custom_color_ranges: str = Form("null"),
    ocr: bool = Query(default_ocr),
    det: bool = Query(default_det),
    old: bool = Query(default_old),
    beta: bool = Query(default_beta),
    use_gpu: bool = Query(default_use_gpu),
    device_id: int = Query(default_device_id),
    show_ad: bool = Query(default_show_ad),
):
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="上传文件为空")
    if len(contents) > CORE_MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail=f"图片大小超过 {CORE_MAX_IMAGE_BYTES // 1024}KB 限制")

    try:
        image = Image.open(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="无法解析上传的图片") from exc

    try:
        colors_list = _ensure_colors_list(json.loads(colors) if colors else [])
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="colors JSON 解析失败") from exc
    custom_ranges = _ensure_custom_ranges(custom_color_ranges)
    probability_flag = _coerce_bool_param(probability, "probability")

    config_key = f"ocr={ocr}-det={det}-old={old}-beta={beta}-gpu={use_gpu}-dev={device_id}"
    ocr_instance = get_ocr_instance(config_key, ocr, det, old, beta, use_gpu, device_id, show_ad, default_import_onnx_path, default_charsets_path)
    start_time = time.time()
    try:
        if probability_flag:
            result = ocr_instance.classification(image, probability=True, colors=colors_list, custom_color_ranges=custom_ranges)
            return {"result": result, "probability": result, "processing_time": time.time() - start_time}
        result = ocr_instance.classification(image, colors=colors_list, custom_color_ranges=custom_ranges)
        return {"result": result, "processing_time": time.time() - start_time}
    except (DdddOcrInputError, InvalidImageError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("OCR文件识别失败: %s", exc)
        raise HTTPException(status_code=500, detail=f"OCR识别失败: {exc}") from exc
    finally:
        try:
            image.close()
        except Exception:
            pass
        background_tasks.add_task(cleanup_inactive_instances)


@app.post("/det", response_model=DetectionResponse)
async def object_detection(
    request: Base64Image,
    background_tasks: BackgroundTasks,
    ocr: bool = Query(False),
    det: bool = Query(True),
    use_gpu: bool = Query(default_use_gpu),
    device_id: int = Query(default_device_id),
    show_ad: bool = Query(default_show_ad),
):
    img_data = _decode_base64_bytes(request.image)
    config_key = f"ocr={ocr}-det={det}-gpu={use_gpu}-dev={device_id}"
    ocr_instance = get_ocr_instance(config_key, ocr, det, False, False, use_gpu, device_id, show_ad, default_import_onnx_path, default_charsets_path)
    start_time = time.time()
    try:
        result = ocr_instance.detection(img_bytes=img_data)
        return {"result": result, "processing_time": time.time() - start_time}
    except (DdddOcrInputError, InvalidImageError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("目标检测失败: %s", exc)
        raise HTTPException(status_code=500, detail=f"目标检测失败: {exc}") from exc
    finally:
        background_tasks.add_task(cleanup_inactive_instances)


@app.post("/det/file", response_model=DetectionResponse)
async def object_detection_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    ocr: bool = Query(False),
    det: bool = Query(True),
    use_gpu: bool = Query(default_use_gpu),
    device_id: int = Query(default_device_id),
    show_ad: bool = Query(default_show_ad),
):
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="上传文件为空")
    if len(contents) > CORE_MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail=f"图片大小超过 {CORE_MAX_IMAGE_BYTES // 1024}KB 限制")

    config_key = f"ocr={ocr}-det={det}-gpu={use_gpu}-dev={device_id}"
    ocr_instance = get_ocr_instance(config_key, ocr, det, False, False, use_gpu, device_id, show_ad, default_import_onnx_path, default_charsets_path)
    start_time = time.time()
    try:
        result = ocr_instance.detection(img_bytes=contents)
        return {"result": result, "processing_time": time.time() - start_time}
    except (DdddOcrInputError, InvalidImageError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("目标检测文件识别失败: %s", exc)
        raise HTTPException(status_code=500, detail=f"目标检测失败: {exc}") from exc
    finally:
        background_tasks.add_task(cleanup_inactive_instances)


@app.post("/slide_match", response_model=SlideMatchResponse)
async def slide_match_recognition(
    request: SlideMatchRequest,
    background_tasks: BackgroundTasks,
    ocr: bool = Query(False),
    det: bool = Query(False),
    use_gpu: bool = Query(default_use_gpu),
    device_id: int = Query(default_device_id),
    show_ad: bool = Query(default_show_ad),
):
    target_data = _decode_base64_bytes(request.target_image)
    background_data = _decode_base64_bytes(request.background_image)
    config_key = f"ocr={ocr}-det={det}-gpu={use_gpu}-dev={device_id}"
    ocr_instance = get_ocr_instance(config_key, ocr, det, False, False, use_gpu, device_id, show_ad, default_import_onnx_path, default_charsets_path)

    start_time = time.time()
    try:
        # 关键修复：DdddOcr.slide_match 的签名是 slide_match(target_img, background_img, simple_target=False)
        # 不能使用 target_bytes/background_bytes/flag 关键字参数。
        result = ocr_instance.slide_match(target_data, background_data, simple_target=request.simple_target)
        result = _normalize_slide_match_result(result)
        return {"result": result, "processing_time": time.time() - start_time}
    except (DdddOcrInputError, InvalidImageError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("滑块匹配失败: %s", exc)
        raise HTTPException(status_code=500, detail=f"滑块匹配失败: {exc}") from exc
    finally:
        background_tasks.add_task(cleanup_inactive_instances)


@app.post("/slide_comparison", response_model=SlideComparisonResponse)
async def slide_comparison_recognition(
    request: SlideComparisonRequest,
    background_tasks: BackgroundTasks,
    ocr: bool = Query(False),
    det: bool = Query(False),
    use_gpu: bool = Query(default_use_gpu),
    device_id: int = Query(default_device_id),
    show_ad: bool = Query(default_show_ad),
):
    target_data = _decode_base64_bytes(request.target_image)
    background_data = _decode_base64_bytes(request.background_image)
    config_key = f"ocr={ocr}-det={det}-gpu={use_gpu}-dev={device_id}"
    ocr_instance = get_ocr_instance(config_key, ocr, det, False, False, use_gpu, device_id, show_ad, default_import_onnx_path, default_charsets_path)

    start_time = time.time()
    try:
        # 关键修复：DdddOcr.slide_comparison 的签名是 slide_comparison(target_img, background_img)
        result = ocr_instance.slide_comparison(target_data, background_data)
        return {"result": result, "processing_time": time.time() - start_time}
    except (DdddOcrInputError, InvalidImageError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("滑块比较失败: %s", exc)
        raise HTTPException(status_code=500, detail=f"滑块比较失败: {exc}") from exc
    finally:
        background_tasks.add_task(cleanup_inactive_instances)


@app.post("/set_charset_range")
async def set_charset_range(
    request: CharsetRangeRequest,
    background_tasks: BackgroundTasks,
    ocr: bool = Query(True),
    det: bool = Query(False),
    old: bool = Query(default_old),
    beta: bool = Query(default_beta),
    use_gpu: bool = Query(default_use_gpu),
    device_id: int = Query(default_device_id),
    show_ad: bool = Query(default_show_ad),
):
    config_key = f"ocr={ocr}-det={det}-old={old}-beta={beta}-gpu={use_gpu}-dev={device_id}"
    ocr_instance = get_ocr_instance(config_key, ocr, det, old, beta, use_gpu, device_id, show_ad, default_import_onnx_path, default_charsets_path)
    start_time = time.time()
    try:
        ocr_instance.set_ranges(request.charset_range)
        return {"result": "字符范围设置成功", "charset_range": request.charset_range, "processing_time": time.time() - start_time}
    except (DdddOcrInputError, InvalidImageError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("设置字符范围失败: %s", exc)
        raise HTTPException(status_code=500, detail=f"设置字符范围失败: {exc}") from exc
    finally:
        background_tasks.add_task(cleanup_inactive_instances)


@app.get("/config")
async def get_current_config():
    return {
        "default_config": {
            "ocr": default_ocr,
            "det": default_det,
            "old": default_old,
            "beta": default_beta,
            "use_gpu": default_use_gpu,
            "device_id": default_device_id,
            "show_ad": default_show_ad,
            "import_onnx_path": default_import_onnx_path,
            "charsets_path": default_charsets_path,
        },
        "active_instances": len(ocr_instances),
        "environment": {"python_version": sys.version, "time": time.strftime("%Y-%m-%d %H:%M:%S")},
    }


def main():
    import uvicorn

    host = os.environ.get("DDDDOCR_HOST", "127.0.0.1")
    port = int(os.environ.get("DDDDOCR_PORT", "8000"))
    workers = int(os.environ.get("DDDDOCR_WORKERS", "1"))
    print(f"启动DdddOcr API服务在 {host}:{port}，工作进程数: {workers}")
    uvicorn.run("ddddocr.api:app", host=host, port=port, workers=workers)


if __name__ == "__main__":
    main()
