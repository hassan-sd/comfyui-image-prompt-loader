"""
Image and Prompt Loader Node for ComfyUI
Author: Hassan-sd
Description: Load images with automatic prompt extraction from text files or EXIF data
"""

import os
import json
import hashlib
import tempfile
import uuid
import numpy as np
import torch
import requests
import re
from PIL import Image, ImageOps, ImageSequence
from PIL.ExifTags import TAGS
import folder_paths
import node_helpers


class ImagePromptLoader:
    """
    A custom ComfyUI node that loads images and extracts prompts from either:
    1. Associated text files (with same name as image) in the original directory
    2. EXIF metadata (specifically from User Comment field)    
    3. Civitai URLs with metadata
    """    
    def __init__(self):
        self._last_civitai_image = None
    
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        
        return {
            "required": {
                "input_mode": (["local_upload", "local_path", "civitai_url"], {"default": "local_upload"}),
                "image": (sorted(files), {"image_upload": True}),
            },
            "optional": {
                "custom_image_path": ("STRING", {"default": "", "multiline": False, "placeholder": "Full path to image file (e.g., D:\\path\\to\\image.jpg)"}),
                "civitai_url": ("STRING", {"default": "", "multiline": False, "placeholder": "Civitai image URL (e.g., https://civitai.com/images/123456)"}),
                "civitai_api_token": ("STRING", {"default": "", "multiline": False, "placeholder": "Civitai API Token (optional, for private images)"}),
            },            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("image", "prompt", "negative_prompt", "seed", "steps", "cfg_scale", "ui_info")
    FUNCTION = "load_image_and_prompt"
    OUTPUT_NODE = True

    def load_image_and_prompt(self, input_mode, image, custom_image_path="", civitai_url="", civitai_api_token="", unique_id=None, extra_pnginfo=None):
        # Check input mode and process accordingly
        if input_mode == "civitai_url":
            if not civitai_url or not civitai_url.strip():
                raise ValueError("Civitai URL is required when input_mode is 'civitai_url'")
            print(f"[ImagePromptLoader] Processing Civitai URL: {civitai_url}")
            
            # Download and process Civitai image, then "upload" it as if it was a local file
            downloaded_filename = self._download_civitai_and_upload(civitai_url.strip(), civitai_api_token.strip(), unique_id)
              # Override the image parameter to use our downloaded file  
            # This helps ComfyUI's preview system recognize the new image
            image = downloaded_filename
            image_path = folder_paths.get_annotated_filepath(downloaded_filename)
            print(f"[ImagePromptLoader] Using downloaded Civitai image as uploaded file: {image_path}")
            
            # Store the filename for the UI to reference
            self._last_civitai_image_filename = downloaded_filename
            
        elif input_mode == "local_path":
            if not custom_image_path or not custom_image_path.strip():
                raise ValueError("Custom image path is required when input_mode is 'local_path'")
            image_path = custom_image_path.strip()
            print(f"[ImagePromptLoader] Using custom image path: {image_path}")
            
            # Validate custom image path exists and is a file (not directory)
            if not os.path.exists(image_path):
                raise ValueError(f"Custom image file not found: {image_path}")
            if os.path.isdir(image_path):
                raise ValueError(f"Custom image path is a directory, not a file: {image_path}")
        
        else:  # input_mode == "local_upload"
            # Use ComfyUI's standard method for uploaded images
            image_path = folder_paths.get_annotated_filepath(image)
            print(f"[ImagePromptLoader] Using ComfyUI uploaded image: {image_path}")
        
        # Load the image directly from the determined path
        img = node_helpers.pillow(Image.open, image_path)

        # Process image frames (similar to built-in LoadImage)
        output_images = []
        w, h = None, None
        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image_data = i.convert("RGB")

            if len(output_images) == 0:
                w = image_data.size[0]
                h = image_data.size[1]

            if image_data.size[0] != w or image_data.size[1] != h:
                continue

            image_array = np.array(image_data).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array)[None,]
            output_images.append(image_tensor)

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
        else:
            output_image = output_images[0]
          # Extract prompts
        prompt = ""
        negative_prompt = ""
        seed = 0
        steps = 0
        cfg_scale = 0.0
        
        print(f"[ImagePromptLoader] Processing image: {image_path}")
        
        # For Civitai images, extract metadata first (this is the primary source)
        if input_mode == "civitai_url" and hasattr(self, '_last_civitai_metadata'):
            civitai_prompt, civitai_negative, civitai_seed, civitai_steps, civitai_cfg = self._extract_from_civitai_metadata(self._last_civitai_metadata)
            print(f"[ImagePromptLoader] Civitai metadata - prompt: '{civitai_prompt}', negative: '{civitai_negative}', seed: {civitai_seed}, steps: {civitai_steps}, cfg: {civitai_cfg}")
            
            if civitai_prompt:
                prompt = civitai_prompt
            if civitai_negative:
                negative_prompt = civitai_negative
            if civitai_seed:
                seed = civitai_seed
            if civitai_steps:
                steps = civitai_steps
            if civitai_cfg:
                cfg_scale = civitai_cfg
        
        # If no prompt found from Civitai metadata, try caption files
        if not prompt:
            caption_prompt, caption_negative = self._extract_from_caption_file(image_path, custom_image_path.strip() if custom_image_path else "")
            print(f"[ImagePromptLoader] Caption extraction - prompt: '{caption_prompt}', negative: '{caption_negative}'")
            if caption_prompt:
                prompt = caption_prompt
            if caption_negative:
                negative_prompt = caption_negative        # If still no prompt found, try EXIF
        if not prompt:
            exif_prompt, exif_negative = self._extract_from_exif(img)
            print(f"[ImagePromptLoader] EXIF extraction - prompt: '{exif_prompt}', negative: '{exif_negative}'")
            if exif_prompt:
                prompt = exif_prompt
            if exif_negative:
                negative_prompt = exif_negative

        print(f"[ImagePromptLoader] Final output - prompt: '{prompt}', negative: '{negative_prompt}', seed: {seed}, steps: {steps}, cfg_scale: {cfg_scale}")
        
        # Set UI info based on input mode
        ui_info = f"civitai_image:{os.path.basename(image_path)}" if input_mode == "civitai_url" else f"local_image:{os.path.basename(image_path)}"
        
        # Prepare the main result tuple
        result = (output_image, prompt, negative_prompt, seed, steps, cfg_scale, ui_info)
          # For Civitai images, return both result and UI update information
        if input_mode == "civitai_url" and hasattr(self, '_last_civitai_image_filename'):
            # Return result with UI information to update the image preview
            # This follows ComfyUI's pattern for updating node UI elements
            ui_data = {
                "images": [
                    {
                        "filename": self._last_civitai_image_filename,
                        "subfolder": "",
                        "type": "input"
                    }
                ]
            }
            return {"result": result, "ui": ui_data}
        
        # For other input modes, return just the result tuple
        return result

    def _extract_from_caption_file(self, image_path, custom_path=""):
        """Extract prompt from associated text file, prioritizing original location over ComfyUI input"""
        image_name = os.path.basename(image_path)
        image_basename = os.path.splitext(image_name)[0]
        image_dir = os.path.dirname(image_path)
        
        print(f"[ImagePromptLoader] Looking for caption files for: {image_path}")
        print(f"[ImagePromptLoader] Image name: {image_name}")
        print(f"[ImagePromptLoader] Image basename: {image_basename}")
        print(f"[ImagePromptLoader] Image directory: {image_dir}")
        
        # List of directories to search for caption files
        search_directories = []
        
        # If using custom path, search in that directory first
        if custom_path:
            custom_dir = os.path.dirname(custom_path)
            search_directories.append(custom_dir)
            print(f"[ImagePromptLoader] Added custom directory: {custom_dir}")
        
        # Add the current image directory (might be ComfyUI input folder)
        if image_dir not in search_directories:
            search_directories.append(image_dir)
          # If this is in ComfyUI's input folder, also search common dataset locations
        comfyui_input_dir = folder_paths.get_input_directory()
        if os.path.normpath(image_dir) == os.path.normpath(comfyui_input_dir):
            print(f"[ImagePromptLoader] Image is in ComfyUI input folder, searching common dataset locations")
            
            # Extract potential folder names from image filename
            # For files like "Minlyva_54026331.jpg" or "Ai_Collector_27036040.jpg"
            potential_folder_names = []
            
            # Method 1: Split by underscore and try different combinations
            parts = image_basename.split('_')
            if len(parts) >= 2:
                # Try first part only (e.g., "Minlyva")
                potential_folder_names.append(parts[0])
                # Try first two parts joined (e.g., "Ai_Collector")
                if len(parts) >= 2:
                    potential_folder_names.append('_'.join(parts[:2]))
                # Try all parts except the last (likely numeric ID)
                if len(parts) > 2:
                    potential_folder_names.append('_'.join(parts[:-1]))
            
            print(f"[ImagePromptLoader] Potential folder names: {potential_folder_names}")
            
            # Common dataset directories where the same image might exist
            base_dirs = [
                "D:\\datasets\\gallery-dl\\gallery-dl\\civitai",
                "D:\\datasets",
                "C:\\Users\\Public\\Downloads",
            ]
            
            # Try each potential folder name
            for folder_name in potential_folder_names:
                for base_dir in base_dirs:
                    potential_dirs = [
                        os.path.join(base_dir, folder_name, "images"),  # e.g., D:\datasets\...\Ai_Collector\images
                        os.path.join(base_dir, folder_name),            # e.g., D:\datasets\...\Ai_Collector
                    ]
                    
                    for potential_dir in potential_dirs:
                        try:
                            if os.path.exists(potential_dir):
                                # Check if the image file exists in this directory
                                potential_image_path = os.path.join(potential_dir, image_name)
                                if os.path.exists(potential_image_path):
                                    search_directories.append(potential_dir)
                                    print(f"[ImagePromptLoader] Found matching image in: {potential_dir}")
                                    break
                        except Exception as e:
                            print(f"[ImagePromptLoader] Error checking directory {potential_dir}: {e}")
                            continue
                    else:
                        continue
                    break  # Exit folder_name loop if we found a match
                else:
                    continue
                break  # Exit base_dirs loop if we found a match
        
        # Remove duplicates while preserving order
        search_directories = list(dict.fromkeys(search_directories))
        print(f"[ImagePromptLoader] Search directories: {search_directories}")
        
        # Try different text file patterns in each directory
        for search_dir in search_directories:
            print(f"[ImagePromptLoader] Searching in directory: {search_dir}")
            
            try:
                text_file_patterns = [
                    os.path.join(search_dir, f"{image_basename}.txt"),     # image.jpg -> image.txt
                    os.path.join(search_dir, f"{image_name}.txt"),         # image.jpg -> image.jpg.txt
                ]
                
                for text_file_path in text_file_patterns:
                    print(f"[ImagePromptLoader] Checking for text file: {text_file_path}")
                    if os.path.exists(text_file_path):
                        print(f"[ImagePromptLoader] Found text file: {text_file_path}")
                        try:
                            with open(text_file_path, 'r', encoding='utf-8') as f:
                                content = f.read().strip()
                                print(f"[ImagePromptLoader] Text file content: {content[:100]}...")
                                return self._parse_caption_content(content)
                        except Exception as e:
                            print(f"Error reading caption file {text_file_path}: {e}")
                            continue
                    else:
                        print(f"[ImagePromptLoader] Text file not found: {text_file_path}")
                        
            except Exception as e:
                print(f"[ImagePromptLoader] Error searching directory {search_dir}: {e}")
                continue
        
        print("[ImagePromptLoader] No caption files found in any search directory")
        return "", ""

    def _parse_caption_content(self, content):
        """Parse caption file content to extract prompt and negative prompt"""
        # Try to parse as JSON first (for more structured formats)
        try:
            data = json.loads(content)
            prompt = data.get('prompt', '')
            negative_prompt = data.get('negative_prompt', data.get('negativePrompt', ''))
            return prompt, negative_prompt
        except json.JSONDecodeError:
            pass
        
        # Parse as plain text with potential negative prompt markers
        lines = content.split('\n')
        prompt_lines = []
        negative_lines = []
        current_section = 'prompt'
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for negative prompt markers
            if line.lower().startswith('negative') or line.lower().startswith('neg:'):
                current_section = 'negative'
                # Remove the marker from the line
                line = line.split(':', 1)[-1].strip()
                if line:
                    negative_lines.append(line)
                continue
            
            if current_section == 'prompt':
                prompt_lines.append(line)
            else:
                negative_lines.append(line)
        
        prompt = ' '.join(prompt_lines)
        negative_prompt = ' '.join(negative_lines)
        
        return prompt, negative_prompt

    def _extract_from_exif(self, img):
        """Extract prompt from EXIF User Comment field"""
        print("[ImagePromptLoader] Extracting from EXIF data")
        try:
            exif_data = img.getexif()
            if not exif_data:
                print("[ImagePromptLoader] No EXIF data found")
                return "", ""
            
            print(f"[ImagePromptLoader] EXIF data found with {len(exif_data)} entries")
            
            # Look for User Comment field (tag 37510 or 0x9286)
            user_comment = None
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                print(f"[ImagePromptLoader] EXIF tag: {tag} ({tag_id}) = {str(value)[:100]}...")
                if tag == "UserComment":
                    user_comment = value
                    print(f"[ImagePromptLoader] Found UserComment: {str(value)[:200]}...")
                    break
            
            if not user_comment:
                print("[ImagePromptLoader] No UserComment field found in EXIF")
                return "", ""
            
            # Decode the user comment (might be bytes or string)
            if isinstance(user_comment, bytes):
                try:
                    # Skip the first 8 bytes which are typically encoding info
                    user_comment = user_comment[8:].decode('utf-8')
                    print(f"[ImagePromptLoader] Decoded UserComment: {user_comment[:200]}...")
                except UnicodeDecodeError:
                    user_comment = user_comment[8:].decode('utf-8', errors='ignore')
                    print(f"[ImagePromptLoader] Decoded UserComment (with errors ignored): {user_comment[:200]}...")
            else:
                print(f"[ImagePromptLoader] UserComment is already string: {user_comment[:200]}...")
            
            # Parse the comment content as JSON
            result = self._parse_exif_comment(user_comment)
            print(f"[ImagePromptLoader] EXIF parsing result: {result}")
            return result
            
        except Exception as e:
            print(f"[ImagePromptLoader] Error extracting EXIF data: {e}")
            return "", ""

    def _parse_exif_comment(self, comment):
        """Parse EXIF comment field which might contain JSON metadata"""
        print(f"[ImagePromptLoader] Parsing EXIF comment: {comment[:200]}...")
        try:
            # Try parsing as JSON first
            data = json.loads(comment)
            print("[ImagePromptLoader] Successfully parsed JSON data")
            print(f"[ImagePromptLoader] JSON keys: {list(data.keys())}")
            
            # Look for extraMetadata field
            if 'extraMetadata' in data:
                print("[ImagePromptLoader] Found extraMetadata field")
                extra_metadata = data['extraMetadata']
                if isinstance(extra_metadata, str):
                    print("[ImagePromptLoader] extraMetadata is string, parsing...")
                    extra_data = json.loads(extra_metadata)
                    prompt = extra_data.get('prompt', '')
                    negative_prompt = extra_data.get('negativePrompt', '')
                    print(f"[ImagePromptLoader] Extracted from extraMetadata - prompt: '{prompt[:100]}...', negative: '{negative_prompt[:100]}...'")
                    return prompt, negative_prompt
              # Direct prompt/negative_prompt fields
            prompt = data.get('prompt', '')
            negative_prompt = data.get('negative_prompt', data.get('negativePrompt', ''))
            print(f"[ImagePromptLoader] Extracted directly - prompt: '{prompt[:100]}...', negative: '{negative_prompt[:100]}...'")
            return prompt, negative_prompt
            
        except json.JSONDecodeError as e:
            print(f"[ImagePromptLoader] JSON decode error: {e}")
            # If not JSON, treat as plain text
            return comment, ""

    def _load_from_civitai(self, civitai_url, api_token="", unique_id=None):
        """Load image and metadata from Civitai URL"""
        try:
            print(f"[ImagePromptLoader] Loading from Civitai URL: {civitai_url}")
            
            # Extract image ID from URL
            image_id = self._extract_civitai_image_id(civitai_url)
            if not image_id:
                raise ValueError(f"Could not extract image ID from URL: {civitai_url}")
            
            print(f"[ImagePromptLoader] Extracted image ID: {image_id}")
              # Get image metadata from Civitai API
            metadata = self._get_civitai_metadata(image_id, api_token)
              # Download the image
            image_url = metadata.get('url')
            if not image_url:
                raise ValueError("No image URL found in Civitai metadata")
            
            print(f"[ImagePromptLoader] Downloading image from: {image_url}")
            img, input_filename = self._download_image(image_url, unique_id)
            
            # Store the input filename for UI updates (if available)
            if input_filename:
                self._last_civitai_image = input_filename
                self.update_preview(input_filename, "", "input")
                print(f"[ImagePromptLoader] Civitai image available as: {input_filename}")
            else:
                self._last_civitai_image = None
            
            # Process image frames (similar to standard image loading)
            output_images = []
            w, h = None, None
            excluded_formats = ['MPO']

            for i in ImageSequence.Iterator(img):
                i = node_helpers.pillow(ImageOps.exif_transpose, i)

                if i.mode == 'I':
                    i = i.point(lambda i: i * (1 / 255))
                image_data = i.convert("RGB")

                if len(output_images) == 0:
                    w = image_data.size[0]
                    h = image_data.size[1]

                if image_data.size[0] != w or image_data.size[1] != h:
                    continue

                image_array = np.array(image_data).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_array)[None,]
                output_images.append(image_tensor)

            if len(output_images) > 1 and img.format not in excluded_formats:
                output_image = torch.cat(output_images, dim=0)
            else:
                output_image = output_images[0]
            
            # Extract prompts from Civitai metadata
            prompt = ""
            negative_prompt = ""
            seed = 0
            steps = 0
            cfg_scale = 0.0
            
            # Try to get prompt from metadata
            civitai_prompt, civitai_negative, civitai_seed, civitai_steps, civitai_cfg = self._extract_from_civitai_metadata(metadata)
            print(f"[ImagePromptLoader] Civitai metadata - prompt: '{civitai_prompt}', negative: '{civitai_negative}', seed: {civitai_seed}, steps: {civitai_steps}, cfg: {civitai_cfg}")
            
            if civitai_prompt:                prompt = civitai_prompt
            if civitai_negative:
                negative_prompt = civitai_negative
            if civitai_seed:
                seed = civitai_seed
            if civitai_steps:
                steps = civitai_steps
            if civitai_cfg:
                cfg_scale = civitai_cfg
            
            # If no metadata prompt found, try EXIF
            if not prompt:
                exif_prompt, exif_negative = self._extract_from_exif(img)
                print(f"[ImagePromptLoader] EXIF extraction - prompt: '{exif_prompt}', negative: '{exif_negative}'")
                if exif_prompt:
                    prompt = exif_prompt
                if exif_negative:
                    negative_prompt = exif_negative
            
            # If still no prompt found, inform user
            if not prompt:
                prompt = "No prompt metadata found for this Civitai image"
                print(f"[ImagePromptLoader] No prompt metadata available for image {image_id}")

            print(f"[ImagePromptLoader] Final Civitai output - prompt: '{prompt}', negative: '{negative_prompt}', seed: {seed}, steps: {steps}, cfg_scale: {cfg_scale}")
            ui_info = f"civitai_image:{input_filename}" if input_filename else "civitai_image:none"
            return (output_image, prompt, negative_prompt, seed, steps, cfg_scale, ui_info)
            
        except Exception as e:
            print(f"[ImagePromptLoader] Error loading from Civitai: {e}")
            # Return empty results on error
            empty_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (empty_tensor, f"Error loading from Civitai: {str(e)}", "", 0, 0, 0.0, "error")

    def _extract_civitai_image_id(self, url):
        """Extract image ID from Civitai URL"""
        # Support various Civitai URL formats for both .com and .green domains
        patterns = [
            r'civitai\.(?:com|green)/images/(\d+)',
            r'civitai\.(?:com|green)/api/v1/images/(\d+)',
            r'/images/(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)        
        return None

    def _get_civitai_metadata(self, image_id, api_token=""):
        """Get image metadata from Civitai API"""
        # Use the correct API endpoint structure
        api_url = f"https://civitai.com/api/v1/images?limit=1&nsfw=true&imageId={image_id}"
        print(f"[ImagePromptLoader] Requesting metadata from: {api_url}")
        
        headers = {
            'User-Agent': 'ComfyUI-ImagePromptLoader/1.0'
        }
        
        # Add API token if provided
        if api_token:
            headers['Authorization'] = f'Bearer {api_token}'
        
        try:
            response = requests.get(api_url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                print(f"[ImagePromptLoader] Received response keys: {list(data.keys())}")
                
                # Check if we have items in the response
                if 'items' in data and len(data['items']) > 0:
                    metadata = data['items'][0]  # Get first (and should be only) item                    print(f"[ImagePromptLoader] Image metadata keys: {list(metadata.keys())}")
                    return metadata
                else:
                    raise Exception("No image data found in API response")
            else:
                response.raise_for_status()
                
        except Exception as e:
            print(f"[ImagePromptLoader] Error with API call: {e}")
            # If API calls fail, try to get basic image info by scraping the page
            try:
                print(f"[ImagePromptLoader] API call failed, trying direct image access...")
                return self._get_image_direct(image_id)
            except Exception as e2:
                print(f"[ImagePromptLoader] Direct access also failed: {e2}")
                raise Exception(f"Could not access image metadata. API error: {str(e)}")

    def _download_image(self, image_url, unique_id=None):
        """Download image from URL and save to input folder with predictable name"""
        headers = {
            'User-Agent': 'ComfyUI-ImagePromptLoader/1.0'
        }
        
        response = requests.get(image_url, headers=headers, timeout=60)
        response.raise_for_status()
        
        from io import BytesIO
        img = node_helpers.pillow(Image.open, BytesIO(response.content))
        
        # Save image to ComfyUI's input folder for preview
        try:
            # Get the input directory from ComfyUI
            input_dir = folder_paths.get_input_directory()
            
            # Create a more predictable filename using node ID if available
            if unique_id:
                base_name = f"civitai_node_{unique_id}"
            else:
                # Fallback to timestamp-based name
                import time
                base_name = f"civitai_{int(time.time())}"
            
            # Try to guess the file extension from the response
            content_type = response.headers.get('content-type', '')
            if 'jpeg' in content_type or 'jpg' in content_type:
                ext = '.jpg'
            elif 'png' in content_type:
                ext = '.png'
            elif 'webp' in content_type:
                ext = '.webp'
            else:
                # Default to jpg if we can't determine
                ext = '.jpg'
            
            input_filename = f"{base_name}{ext}"
            input_path = os.path.join(input_dir, input_filename)
            
            # Remove existing file if it exists
            if os.path.exists(input_path):
                os.remove(input_path)
              # Save the image
            img.save(input_path, optimize=True, quality=95)
            print(f"[ImagePromptLoader] Saved Civitai image to input folder: {input_path}")
            
            return img, input_filename
            
        except Exception as e:
            print(f"[ImagePromptLoader] Warning: Could not save image to input folder: {e}")
            # Return img with None filename if saving fails
            return img, None

    def _download_civitai_and_upload(self, civitai_url, api_token="", unique_id=None):
        """
        Downloads an image from Civitai, saves it to the input folder with a predictable filename,
        and returns the filename so it can be used as if it was uploaded by the user.
        Also stores the Civitai metadata for later prompt extraction.
        """
        try:
            # Extract image ID from Civitai URL
            image_id = self._extract_civitai_image_id(civitai_url)
            if not image_id:
                raise ValueError(f"Could not extract image ID from Civitai URL: {civitai_url}")
            
            print(f"[ImagePromptLoader] Downloading Civitai image ID: {image_id}")
            
            # Get metadata first (this is the primary source for prompts)
            metadata = self._get_civitai_metadata(image_id, api_token)
            
            # Store metadata immediately for prompt extraction
            self._last_civitai_metadata = metadata
            
            # Find the image URL from metadata
            image_url = None
            if metadata and 'url' in metadata:
                image_url = metadata['url']
            elif metadata and 'images' in metadata and len(metadata['images']) > 0:
                # Sometimes the metadata structure is different
                image_url = metadata['images'][0].get('url')
            
            if not image_url:
                # Fallback: try to get direct image URL
                print(f"[ImagePromptLoader] No image URL in metadata, trying direct access")
                direct_metadata = self._get_image_direct(image_id)
                image_url = direct_metadata.get('url')
                if not image_url:
                    raise ValueError(f"Could not get image URL from Civitai ID: {image_id}")
            
            # Download image from the URL using requests directly (simpler approach)
            print(f"[ImagePromptLoader] Downloading image from: {image_url}")
            headers = {'User-Agent': 'ComfyUI-ImagePromptLoader/1.0'}
            response = requests.get(image_url, headers=headers, timeout=60)
            response.raise_for_status()
            
            # Save to input folder with predictable filename
            input_dir = folder_paths.get_input_directory()
            filename = f"civitai_{image_id}.jpg"
            input_path = os.path.join(input_dir, filename)
            
            # Save the downloaded image
            with open(input_path, 'wb') as f:
                f.write(response.content)
            print(f"[ImagePromptLoader] Saved Civitai image to input folder: {input_path}")
            
            return filename
            
        except Exception as e:
            print(f"[ImagePromptLoader] Error downloading from Civitai: {e}")
            raise ValueError(f"Failed to download image from Civitai: {e}")

    def _extract_from_civitai_metadata(self, metadata):
        """Extract prompt, negative prompt, and generation parameters from Civitai metadata"""
        prompt = ""
        negative_prompt = ""
        seed = 0
        steps = 0
        cfg_scale = 0.0
        
        try:
            # Check meta field for generation parameters
            meta = metadata.get('meta', {})
            if meta:
                print(f"[ImagePromptLoader] Found meta field with keys: {list(meta.keys())}")
                
                # Extract prompt
                if 'prompt' in meta:
                    prompt = str(meta['prompt'])
                
                # Extract negative prompt (try different possible field names)
                for neg_field in ['negativePrompt', 'negative_prompt', 'Negative prompt']:
                    if neg_field in meta and meta[neg_field]:
                        negative_prompt = str(meta[neg_field])
                        break
                
                # Extract generation parameters
                if 'seed' in meta:
                    try:
                        seed = int(meta['seed'])
                    except (ValueError, TypeError):
                        seed = 0
                
                if 'steps' in meta:
                    try:
                        steps = int(meta['steps'])
                    except (ValueError, TypeError):
                        steps = 0
                  # CFG Scale might be named differently
                for cfg_field in ['cfgScale', 'cfg_scale', 'CFG Scale', 'guidance_scale']:
                    if cfg_field in meta and meta[cfg_field]:
                        try:
                            cfg_scale = float(meta[cfg_field])
                            break
                        except (ValueError, TypeError):
                            continue
            
            print(f"[ImagePromptLoader] Extracted from Civitai - prompt: '{prompt[:100]}...', negative: '{negative_prompt[:100]}...', seed: {seed}, steps: {steps}, cfg: {cfg_scale}")
            
        except Exception as e:
            print(f"[ImagePromptLoader] Error extracting from Civitai metadata: {e}")        
        return prompt, negative_prompt, seed, steps, cfg_scale
    
    @classmethod
    def IS_CHANGED(cls, input_mode, image, custom_image_path="", civitai_url="", civitai_api_token="", **kwargs):
        # Handle cache key based on input mode
        if input_mode == "civitai_url":
            if civitai_url and civitai_url.strip():
                m = hashlib.sha256()
                m.update(civitai_url.strip().encode('utf-8'))
                # Include token in cache key if provided (but don't log it)
                if civitai_api_token and civitai_api_token.strip():
                    m.update(civitai_api_token.strip().encode('utf-8'))
                return m.digest().hex()
            return ""
            
        # Determine the image path based on input mode
        if input_mode == "local_path":
            if not custom_image_path or not custom_image_path.strip():
                return ""
            image_path = custom_image_path.strip()
        else:  # input_mode == "local_upload"
            image_path = folder_paths.get_annotated_filepath(image)
        
        # Check if image file has changed
        if not os.path.exists(image_path):
            return ""
        
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        
        # Also check text file (always enabled now)
        image_name = os.path.basename(image_path)
        image_basename = os.path.splitext(image_name)[0]
        image_dir = os.path.dirname(image_path)
        
        text_file_patterns = [
            os.path.join(image_dir, f"{image_basename}.txt"),
            os.path.join(image_dir, f"{image_name}.txt"),
        ]
        for text_file_path in text_file_patterns:
            if os.path.exists(text_file_path):
                with open(text_file_path, 'rb') as f:
                    m.update(f.read())
                break
        
        return m.digest().hex()

    def get_civitai_image_info(self):
        """Return info for the last downloaded Civitai image for UI updates"""
        if hasattr(self, '_last_civitai_image_filename'):
            return {
                "filename": self._last_civitai_image_filename,
                "subfolder": "",
                "type": "input"
            }
        return None

    @classmethod
    def VALIDATE_INPUTS(cls, input_mode, image, custom_image_path="", civitai_url="", civitai_api_token=""):
        # Validate based on input mode
        if input_mode == "civitai_url":
            if not civitai_url or not civitai_url.strip():
                return "Civitai URL is required when input_mode is 'civitai_url'"
            url = civitai_url.strip()
            if not (url.startswith('http://') or url.startswith('https://')):
                return f"Invalid Civitai URL format: {url}"
            if not ('civitai.com' in url or 'civitai.green' in url):
                return f"URL must be from civitai.com or civitai.green: {url}"
            return True
            
        elif input_mode == "local_path":
            if not custom_image_path or not custom_image_path.strip():
                return "Custom image path is required when input_mode is 'local_path'"
            custom_path = custom_image_path.strip()
            if not os.path.exists(custom_path):
                return f"Invalid custom image file: {custom_path}"
            if os.path.isdir(custom_path):
                return f"Custom image path is a directory, not a file: {custom_path}"
            return True
            
        else:  # input_mode == "local_upload"
            if not folder_paths.exists_annotated_filepath(image):
                return f"Invalid image file: {image}"
            return True

    def _get_image_direct(self, image_id):
        """Try to get image URL directly when API fails"""
        # Try to construct direct image URL patterns that Civitai uses
        possible_urls = [
            f"https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/{image_id}/original=true,quality=90/",
            f"https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/{image_id}/",
            f"https://civitai.com/images/{image_id}",
        ]
        
        headers = {
            'User-Agent': 'ComfyUI-ImagePromptLoader/1.0'
        }
        
        # Try each possible URL to see if we can access the image
        for url in possible_urls:
            try:
                print(f"[ImagePromptLoader] Trying direct image URL: {url}")
                response = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
                if response.status_code == 200:
                    # Check if it's actually an image
                    content_type = response.headers.get('content-type', '')
                    if 'image' in content_type:
                        print(f"[ImagePromptLoader] Found direct image at: {url}")
                        return {'url': url, 'meta': {}}
            except Exception as e:
                print(f"[ImagePromptLoader] Failed to access {url}: {e}")
                continue
        
        # If no direct URL works, return a generic error
        raise Exception("Could not find accessible image URL")


    def get_last_civitai_image(self):
        """Get the filename of the last downloaded Civitai image for UI updates"""
        return getattr(self, '_last_civitai_image', None)
    
    def OUTPUT_NODE(self):
        """Mark this as an output node so it can send data to the UI"""
        return True
    
    def update_preview(self, filename, subfolder="", folder_type="input"):
        """Update the node preview with a new image"""
        if filename and hasattr(self, 'last_preview_update'):
            self.last_preview_update = {
                "filename": filename,
                "subfolder": subfolder,
                "type": folder_type
            }

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ImagePromptLoader": ImagePromptLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImagePromptLoader": "Image & Prompt Loader"
}
