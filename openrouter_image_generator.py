import base64
import requests
import logging

log = logging.getLogger(__name__)


class OpenRouterImageGenerator:
    """Client for generating images using the OpenRouter API."""
    
    def __init__(self, api_key: str, model: str = "google/gemini-2.5-flash-image-preview"):
        """
        Initialize the OpenRouter image generator.
        
        Args:
            api_key: OpenRouter API key
            model: Model to use for image generation. Default is google/gemini-2.5-flash-image-preview
                   Other options: black-forest-labs/flux.2-pro, black-forest-labs/flux.2-flex
        """
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def generate_image(
        self, 
        prompt: str, 
        size: str = "512x512",
        aspect_ratio: str = None,
        image_size: str = None
    ) -> bytes:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: Text description of the image to generate
            size: Image size in format "WxH" (e.g., "512x512", "1024x1024")
                  This is used to determine aspect_ratio if not explicitly provided
            aspect_ratio: Explicit aspect ratio (e.g., "1:1", "16:9", "4:3")
                         Supported: 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9
            image_size: Image resolution (Gemini only): "1K", "2K", or "4K"
        
        Returns:
            Image data as bytes (PNG format)
        """
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "modalities": ["image", "text"]
        }
        
        # Add image_config if aspect_ratio or image_size is specified
        image_config = {}
        
        # Determine aspect ratio from size parameter if not explicitly provided
        if aspect_ratio:
            image_config["aspect_ratio"] = aspect_ratio
        elif size:
            try:
                width, height = map(int, size.lower().split("x"))
                # Map common sizes to aspect ratios
                ratio = width / height
                if abs(ratio - 1.0) < 0.1:  # Square
                    image_config["aspect_ratio"] = "1:1"
                elif abs(ratio - 16/9) < 0.1:
                    image_config["aspect_ratio"] = "16:9"
                elif abs(ratio - 9/16) < 0.1:
                    image_config["aspect_ratio"] = "9:16"
                elif abs(ratio - 4/3) < 0.1:
                    image_config["aspect_ratio"] = "4:3"
                elif abs(ratio - 3/4) < 0.1:
                    image_config["aspect_ratio"] = "3:4"
                # Add more mappings as needed
            except Exception as e:
                log.warning(f"Could not parse size '{size}', using default: {e}")
        
        if image_size:
            image_config["image_size"] = image_size
            
        if image_config:
            payload["image_config"] = image_config
        
        log.info(f"Generating image with prompt: '{prompt[:100]}...' using model {self.model}")
        
        response = requests.post(self.base_url, headers=headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f'OpenRouter API Error: {response.status_code} - {response.text}')
        
        result = response.json()
        
        # Extract the generated image from the response
        if result.get("choices"):
            message = result["choices"][0]["message"]
            if message.get("images"):
                # Get the first image's data URL
                image_data_url = message["images"][0]["image_url"]["url"]
                
                # Parse base64 data URL (format: data:image/png;base64,<data>)
                if image_data_url.startswith("data:image/"):
                    base64_data = image_data_url.split(",", 1)[1]
                    image_bytes = base64.b64decode(base64_data)
                    log.info(f"Successfully generated image ({len(image_bytes)} bytes)")
                    return image_bytes
                else:
                    raise Exception(f"Unexpected image URL format: {image_data_url[:50]}")
            else:
                raise Exception("No images found in response")
        else:
            raise Exception(f"No choices in response: {result}")