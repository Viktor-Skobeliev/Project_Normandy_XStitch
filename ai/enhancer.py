"""STEP 13 — Local AI Multi-Agent Audit: DataCritic + VisionAnalyst."""

from __future__ import annotations

import json
import os
import re
from PIL import Image
from typing import Any

from ai.vision_agent import VisionAgent
from ai.prompts import (
    CRITIC_SYSTEM_PROMPT, build_critic_prompt,
    VISION_SYSTEM_PROMPT, build_vision_prompt,
)
from utils.logger import get_logger

log = get_logger(__name__)


_llama_instance = None


def _get_llama():
    global _llama_instance
    if _llama_instance is None:
        from llama_cpp import Llama
        model_path = os.path.join(os.getcwd(), "models", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Llama model not found: {model_path}")
        log.info("Loading Llama 3.1 (shared instance)...")
        _llama_instance = Llama(
            model_path=model_path,
            n_ctx=3072,
            n_threads=4,
            verbose=False,
        )
        log.info("Llama 3.1 loaded.")
    return _llama_instance


def _llm_call(system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
    """Single LLM call with error handling."""
    llm = _get_llama()
    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.error("LLM call failed: %s", e)
        return f"ERROR: {e}"


def _parse_agent2_response(response: str) -> dict:
    """Parse Agent 2 structured response into a dict."""
    result = {
        "final_verdict": "UNKNOWN",
        "remove_colors": [],
        "scale_factor": 1.0,
        "pdf_summary": "",
    }


    m = re.search(r"FINAL_VERDICT:\s*(PASS|WARNING|FAIL)", response, re.IGNORECASE)
    if m:
        result["final_verdict"] = m.group(1).upper()


    m = re.search(r"REMOVE_COLORS:\s*(.+)", response, re.IGNORECASE)
    if m:
        val = m.group(1).strip()
        if val.lower() != "none":
            result["remove_colors"] = [c.strip() for c in val.split(",") if c.strip()]


    m = re.search(r"SCALE_FACTOR:\s*([0-9.]+)", response, re.IGNORECASE)
    if m:
        try:
            result["scale_factor"] = float(m.group(1))
        except ValueError:
            pass


    m = re.search(r"PDF_SUMMARY:\s*\n?([\s\S]+)", response, re.IGNORECASE)
    if m:
        result["pdf_summary"] = m.group(1).strip()

    return result


def run_local_ai_orchestrator(ctx) -> dict | None:
    """
    Two-agent AI audit pipeline:
      Agent 1 (DataCritic)    — analyzes numbers, finds anomalies
      Agent 2 (VisionAnalyst) — analyzes image metrics + Agent 1 report, produces final verdict
    """
    ctx.report_progress(13, "AI Audit: Step 1/2 — Data Critic analyzing pattern...")
    log.info("AI Audit: starting two-agent pipeline")

    try:

        temp_dir = os.path.join(os.getcwd(), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_preview_path = os.path.join(temp_dir, "ai_preview.png")



        source_arr = ctx.repaired_image if ctx.repaired_image is not None else ctx.quantized_image
        if source_arr is not None:
            img = Image.fromarray(source_arr)
            img.thumbnail((800, 800), Image.LANCZOS)  # resize for speed
            img.save(temp_preview_path)
            src_name = "repaired_image" if ctx.repaired_image is not None else "quantized_image"
            log.info("AI Audit: preview saved (%dx%d) from %s", img.width, img.height, src_name)
        else:
            log.warning("AI Audit: no source image, skipping")
            return None


        ctx.report_progress(13, "AI Audit: Step 1/2 — Vision extracting image metrics...")
        vision = VisionAgent()
        image_metrics = vision.analyze_image(temp_preview_path)


        ctx.report_progress(13, "AI Audit: Step 1/2 — Data Critic analyzing thread data...")
        critic_prompt = build_critic_prompt(
            thread_usage=ctx.thread_usage,
            grid={"width": ctx.grid_width, "height": ctx.grid_height},
            canvas_count=ctx.canvas_count,
            image_metrics=image_metrics,
        )
        log.info("AI Audit: Agent 1 (DataCritic) running...")
        critic_report = _llm_call(CRITIC_SYSTEM_PROMPT, critic_prompt, max_tokens=400)
        log.info("AI Audit: Agent 1 complete.\n%s", critic_report)


        ctx.report_progress(13, "AI Audit: Step 2/2 — Vision Analyst optimizing...")
        vision_prompt = build_vision_prompt(
            image_metrics=image_metrics,
            critic_report=critic_report,
            thread_usage=ctx.thread_usage,
        )
        log.info("AI Audit: Agent 2 (VisionAnalyst) running...")
        vision_report = _llm_call(VISION_SYSTEM_PROMPT, vision_prompt, max_tokens=512)
        log.info("AI Audit: Agent 2 complete.\n%s", vision_report)


        parsed = _parse_agent2_response(vision_report)
        log.info(
            "AI Audit: verdict=%s remove=%s scale=%.2f",
            parsed["final_verdict"], parsed["remove_colors"], parsed["scale_factor"]
        )


        scale = parsed["scale_factor"]
        optimized_usage = {}
        for code, color in ctx.thread_usage.items():
            if code in parsed["remove_colors"]:
                log.info("AI Audit: removing color %s from shopping list", code)
                continue
            if scale != 1.0:
                from core.context import ThreadColor
                from core.thread_calculator import METERS_PER_SKEIN
                new_meters = round(color.meters_needed * scale, 2)
                new_skeins = round(new_meters / METERS_PER_SKEIN, 2)
                optimized_usage[code] = ThreadColor(
                    brand=color.brand,
                    code=color.code,
                    name=color.name,
                    rgb=color.rgb,
                    meters_needed=new_meters,
                    skeins_needed=new_skeins,
                )
            else:
                optimized_usage[code] = color


        ctx.thread_usage = optimized_usage
        ctx.ai_suggestions = {
            "verdict": parsed["final_verdict"],
            "critic_report": critic_report,
            "vision_report": vision_report,
            "removed_colors": parsed["remove_colors"],
            "scale_factor": scale,
            "pdf_summary": parsed["pdf_summary"],
            "image_metrics": image_metrics,
        }

        log.info("AI Audit: pipeline complete. Verdict: %s", parsed["final_verdict"])
        return ctx.ai_suggestions

    except FileNotFoundError as e:
        log.warning("AI Audit skipped — model not found: %s", e)
        return None
    except Exception as e:
        log.error("AI Audit failed: %s", e)
        return None


def run_ai_audit_sync(ctx) -> dict | None:
    return run_local_ai_orchestrator(ctx)
