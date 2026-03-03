"""PROMPTS — Instructions for Local AI Agents (Llama 3.1)."""




def _classify_image(image_metrics: dict) -> str:
    brightness = image_metrics.get("mean_brightness", 128)
    contrast = image_metrics.get("color_std", 0)
    edge_density = image_metrics.get("edge_density_pct", 0)
    unique_colors = image_metrics.get("unique_colors", 0)
    fg = image_metrics.get("fg_coverage_pct", 0)
    tags = []
    if brightness > 200:
        tags.append("high-key / light background")
    elif brightness < 80:
        tags.append("dark / low-key image")
    else:
        tags.append("mid-tone image")
    if contrast > 60:
        tags.append("high contrast")
    elif contrast < 25:
        tags.append("low contrast")
    if edge_density > 15:
        tags.append("geometric / detailed linework")
    elif edge_density < 5:
        tags.append("smooth / painterly")
    if unique_colors < 20:
        tags.append("limited palette")
    elif unique_colors > 200:
        tags.append("photo-realistic complexity")
    if fg < 25:
        tags.append("sparse foreground")
    elif fg > 70:
        tags.append("dense fill")
    return ", ".join(tags) if tags else "standard pattern"


CRITIC_SYSTEM_PROMPT = """You are a Professional Cross-Stitch Technical Auditor.
Your job is to analyze numerical data from a cross-stitch pattern generator and identify anomalies.
Be concise. Use bullet points. Flag real problems only."""


def build_critic_prompt(thread_usage: dict, grid: dict, canvas_count: int, image_metrics: dict) -> str:
    lines = []
    for code, color in thread_usage.items():
        d = color.model_dump() if hasattr(color, 'model_dump') else color
        lines.append(
            f"  {d.get('brand','?')} {code} '{d.get('name','?')}': "
            f"{d.get('meters_needed',0):.1f}m / {d.get('skeins_needed',0):.2f} skeins"
        )
    thread_block = "\n".join(lines) if lines else "  (no data)"
    grid_w = grid.get('width', 0)
    grid_h = grid.get('height', 0)
    total_stitches = grid_w * grid_h
    fg_pct = image_metrics.get('fg_coverage_pct', 0)
    fg_stitches = total_stitches * fg_pct / 100
    expected_meters = fg_stitches * 0.35
    total_meters = sum(
        (color.meters_needed if hasattr(color, 'meters_needed') else color.get('meters_needed', 0))
        for color in thread_usage.values()
    )
    ratio = (total_meters / expected_meters) if expected_meters > 0 else 1.0
    image_type = _classify_image(image_metrics)

    return f"""AUDIT TASK — Cross-Stitch Pattern Analysis

== Pattern Parameters ==
Canvas: {canvas_count} ct (Aida)
Grid size: {grid_w} x {grid_h} stitches ({total_stitches:,} total)
Foreground stitches (estimated): {int(fg_stitches):,}
Expected thread range: up to {expected_meters:.0f}m
Total calculated thread: {total_meters:.1f}m
Usage ratio (actual/expected): {ratio:.2f}

Image type: {image_type}
Active colors: {len(thread_usage)}

== Calculated Thread Usage ==
{thread_block}

== Your Task ==
1. Thread amount check:
   - If ratio > 1.8: flag as OVERESTIMATED
   - If ratio < 0.3 and fg > 20%: flag as UNDERESTIMATED
   - Otherwise: PASS
2. Duplicate detection: flag only if two colors share similar name AND similar meter count (within 10%)
3. Flag colors with <1m as REMOVABLE (not worth buying a skein)
4. Do NOT flag 3-8 colors as a problem — that is normal for most patterns

Return your analysis in this format:
VERDICT: [PASS|WARNING|FAIL]
ISSUES:
- [issue 1 or 'none']
RECOMMENDATIONS:
- [rec 1 or 'none']"""


VISION_SYSTEM_PROMPT = """You are a Senior Cross-Stitch Pattern Optimizer.
You receive image analysis data and a data critic's report.
Your job is to produce final optimization decisions and a human-readable summary in English.
Be specific and actionable."""


def build_vision_prompt(image_metrics: dict, critic_report: str, thread_usage: dict) -> str:
    removable = [
        code for code, color in thread_usage.items()
        if (color.meters_needed if hasattr(color, 'meters_needed') else color.get('meters_needed', 999)) < 1.0
    ]
    total_meters = sum(
        (color.meters_needed if hasattr(color, 'meters_needed') else color.get('meters_needed', 0))
        for color in thread_usage.values()
    )
    image_type = _classify_image(image_metrics)
    notes = []
    brightness = image_metrics.get("mean_brightness", 128)
    contrast = image_metrics.get("color_std", 0)
    fg = image_metrics.get("fg_coverage_pct", 0)
    if brightness > 210:
        notes.append("Image is very bright — white/light areas may inflate foreground coverage reading.")
    if contrast < 20:
        notes.append("Low contrast image — color boundaries may be blurry; thread estimates less precise.")
    if fg < 20:
        notes.append("Sparse foreground — large empty areas mean thread totals should be low.")
    if fg > 75:
        notes.append("Dense fill pattern — high thread usage is expected and normal.")
    context_block = "\n".join(f"- {n}" for n in notes) if notes else "- No special conditions detected."

    return f"""OPTIMIZATION TASK

== Image Analysis ==
Image type: {image_type}
Foreground fill: {fg:.1f}%
Empty background: {100 - fg:.1f}%
Color complexity: {image_metrics.get('unique_colors', 0)} unique colors detected
Mean brightness: {brightness:.0f}/255
Contrast (std): {contrast:.1f}
Edge density: {image_metrics.get('edge_density_pct', 0):.1f}%

== Contextual Notes ==
{context_block}

== Data Critic Report ==
{critic_report}

== Thread Summary ==
Total colors: {len(thread_usage)}
Total meters: {total_meters:.1f}m
Colors with <1m (candidates for removal): {removable if removable else 'none'}

== Your Task ==
1. Based on critic report and image data — confirm or override the verdict.
2. List which colors (if any) should be removed from the shopping list.
3. If thread is overestimated, set SCALE_FACTOR < 1.0 (e.g. 0.75). If correct, use 1.0.
4. Write a SHORT SUMMARY in English for the PDF report (3-5 sentences).
   Tell the crafter: pattern complexity, number of colors, estimated thread, any warnings.

Return in this EXACT format (no extra text before FINAL_VERDICT):
FINAL_VERDICT: [PASS|WARNING|FAIL]
REMOVE_COLORS: [comma-separated codes or 'none']
SCALE_FACTOR: [1.0 or adjusted value]
PDF_SUMMARY:
[your 3-5 sentence summary in English]"""