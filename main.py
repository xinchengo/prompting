import base64
import io
import json
import os
from datetime import datetime
from random import sample
from typing import Dict, Optional
# For structured output
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw, ImageFont
import hydra
import logging

# LangChain
from langchain.chat_models import init_chat_model
from langgraph.func import entrypoint, task

# OpenRouter image generation
from openrouter_image_generator import OpenRouterImageGenerator

load_dotenv()  # Load environment variables from .env file
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# Global handles (initialized in main)
# For structured output

class RenderSettings(BaseModel):
    initial_position: Dict[str, float] = Field(..., description="Initial position of the object")
    initial_velocity: Dict[str, float] = Field(..., description="Initial velocity of the object")
    acceleration: Dict[str, float] = Field(..., description="Acceleration of the object")
    angular_velocity: float = Field(..., description="Angular velocity of the object")

class DescriberOutput(BaseModel):
    background_prompt: str = Field(..., description="Prompt for background image generation")
    foreground_prompt: str = Field(..., description="Prompt for foreground object with green background")
    render_settings: RenderSettings = Field(..., description="Render settings for the object")
    
describer_model = None
reviewer_model = None
image_generator_model = None
image_generator_client: Optional[OpenRouterImageGenerator] = None


def _sanitize(text: str) -> str:
    return text.lower().replace(" ", "-")


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _decode_json_response(raw_text: str) -> Dict:
    """Best-effort JSON parsing from model output."""
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and start < end:
            return json.loads(raw_text[start : end + 1])
        raise


@task
def remove_background(image_bytes: bytes) -> bytes:
    """Simple chroma key to remove green background from a PNG/JPEG bytes payload."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    data = []
    for r, g, b, a in image.getdata():
        if g > max(r, b) + 40 and g > 80:  # treat strong green as background
            data.append((0, 0, 0, 0))
        else:
            data.append((r, g, b, a))
    image.putdata(data)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


@task
def generate_image(prompt: str, size: str = "512x512") -> bytes:
    """Generate an image using OpenRouter API."""
    if image_generator_client is None:
        raise RuntimeError("Image generator client not initialized.")
    
    try:
        return image_generator_client.generate_image(prompt, size=size)
    except Exception as e:
        log.error(f"Failed to generate image: {e}")
        raise


@task
def review_scene(params: DictConfig, scene_plan: Dict) -> str:
    """Use the reviewer model to validate prompts and render settings."""
    if params.options.generation.dry_run:
        log.info("[DRY RUN] Skipping LLM call for review_scene")
        return "[DRY RUN] Approved"
    
    if reviewer_model is None:
        raise RuntimeError("Reviewer model not initialized.")
    
    reviewer_prompt = params.prompts[1].reviewer
    # Convert Pydantic model to dict for JSON serialization
    scene_plan_dict = scene_plan.model_dump() if hasattr(scene_plan, 'model_dump') else scene_plan
    review = reviewer_model.invoke(
        [
            ("system", reviewer_prompt),
            ("user", json.dumps(scene_plan_dict, indent=2)),
        ]
    )
    review_text = review.content if hasattr(review, "content") else str(review)
    return review_text


@task
def describe_scene(params: DictConfig) -> Dict:
    """Use the describer model to propose prompts and render settings."""
    if params.options.generation.dry_run:
        log.info("[DRY RUN] Skipping LLM call for describe_scene")
        return {
            "background_prompt": f"[DRY RUN] Background for {params.motion} in {params.scene_sp}",
            "foreground_prompt": f"[DRY RUN] Foreground object for {params.motion} with green background",
            "render_settings": {
                "initial_position": {"x": 100, "y": 100},
                "initial_velocity": {"vx": 50, "vy": 0},
                "acceleration": {"ax": 0, "ay": 98},
                "angular_velocity": 0
            }
        }
    
    if describer_model is None:
        raise RuntimeError("Describer model not initialized.")

    system_prompt = params.prompts[0].describer
    
    request = (
        f"Motion: {params.motion}\n"
        f"Scene: {params.scene_sp}\n"
        f"Video width: {params.options.video.width}, height: {params.options.video.height}, duration: {params.options.video.duration}s.\n"
        # "Return JSON with keys: background_prompt (string), foreground_prompt (string with green background), "
        # "render_settings (object with initial_position {x,y}, initial_velocity {vx,vy}, acceleration {ax,ay}, angular_velocity w)."
    )
    reply = describer_model.invoke(
        [
            ("system", system_prompt),
            ("user", request),
        ]
    )
    # with_structured_output already returns a DescriberOutput instance
    return reply


@entrypoint()
def create_scene_assets(params: DictConfig):
    log.info(f"Generating for motion: {params.motion}, scene specification: {params.scene_sp}")

    # Step 1: Describe the scene to get prompts and render settings
    scene_plan = describe_scene(params).result()

    if params.options.generation.review:
        review_text = review_scene(params, scene_plan).result()
        if "approved" not in review_text.lower():
            log.warning("Reviewer flagged issues: %s", review_text)
        else:
            log.info("Reviewer approved prompts.")

    # Step 2: Generate background and foreground images
    background_bytes = generate_image(scene_plan.background_prompt).result()
    foreground_bytes = generate_image(scene_plan.foreground_prompt).result()
    foreground_alpha = remove_background(foreground_bytes).result()

    # Step 3: Save outputs with structured filenames
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{_sanitize(params.motion)}__{_sanitize(params.scene_sp)}__{stamp}"
    # out_dir = _ensure_dir(os.path.join(os.getcwd(), "outputs"))
    
    out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    bg_path = os.path.join(out_dir, f"{prefix}_background.png")
    fg_path = os.path.join(out_dir, f"{prefix}_foreground_raw.png")
    fg_clean_path = os.path.join(out_dir, f"{prefix}_foreground_alpha.png")

    with open(bg_path, "wb") as f:
        f.write(background_bytes)
    with open(fg_path, "wb") as f:
        f.write(foreground_bytes)
    with open(fg_clean_path, "wb") as f:
        f.write(foreground_alpha)

    return {
        "background": bg_path,
        "foreground_raw": fg_path,
        "foreground": fg_clean_path,
        "render_settings": scene_plan.render_settings.model_dump() if hasattr(scene_plan.render_settings, 'model_dump') else scene_plan.render_settings,
    }


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    assert cfg.options.generation.policy in ["random", "grid"]

    # Initialize models
    global describer_model, reviewer_model, image_generator_model, image_generator_client
    
    if cfg.options.generation.dry_run:
        log.info("[DRY RUN MODE] Skipping LLM model initialization")
    else:
        describer_model = init_chat_model(
            model=cfg.options.generation.describer_model,
            model_provider=cfg.options.generation.model_provider,
        )
        describer_model = describer_model.with_structured_output(DescriberOutput)
        
        reviewer_model = init_chat_model(
            model=cfg.options.generation.reviewer_model,
            model_provider=cfg.options.generation.model_provider,
        )
        
        image_generator_model = cfg.options.generation.image_generator_model
        
        # Initialize OpenRouter image generator
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
        
        image_generator_client = OpenRouterImageGenerator(
            api_key=openrouter_api_key,
            model=image_generator_model
        )
        
        log.info("Models initialized successfully.")

    # Generate images based on policy
    if cfg.options.generation.policy == "random":
        for _ in range(cfg.options.generation.num_samples):
            motion = sample(cfg.motions, 1)[0]
            scene_sp = sample(cfg.scene_specifications, 1)[0]
            log.info(f"Generating for motion: {motion}, scene specification: {scene_sp}")
            create_scene_assets.invoke(
                OmegaConf.create(
                    {
                        "motion": motion,
                        "scene_sp": scene_sp,
                        "options": cfg.options,
                        "prompts": cfg.prompts,
                    }
                )
            )

    elif cfg.options.generation.policy == "grid":
        for motion in cfg.motions:
            for scene_sp in cfg.scene_specifications:
                log.info(f"Generating for motion: {motion}, scene specification: {scene_sp}")
                create_scene_assets.invoke(
                    OmegaConf.create(
                        {
                            "motion": motion,
                            "scene_sp": scene_sp,
                            "options": cfg.options,
                            "prompts": cfg.prompts,
                        }
                    )
                )


if __name__ == "__main__":
    main()