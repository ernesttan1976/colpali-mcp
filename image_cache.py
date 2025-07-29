import os
import hashlib
from pathlib import Path
import pickle
from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('image_cache')

# Define cache directory
CACHE_DIR = "./data/image_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_file_hash(file_content):
    """Generate a unique hash for file content to use as cache key."""
    return hashlib.md5(file_content).hexdigest()

def get_file_content(file_obj):
    """Extract raw file content from various file object types."""
    if hasattr(file_obj, "read"):
        # File-like object
        pos = file_obj.tell()
        file_obj.seek(0)
        content = file_obj.read()
        file_obj.seek(pos)  # Reset file position
        return content
    elif hasattr(file_obj, "name") and os.path.isfile(file_obj.name):
        # File object with name attribute
        with open(file_obj.name, "rb") as f:
            return f.read()
    elif isinstance(file_obj, str) and os.path.isfile(file_obj):
        # File path as string
        with open(file_obj, "rb") as f:
            return f.read()
    elif isinstance(file_obj, tuple) and len(file_obj) == 2:
        # Tuple of (name, file-like object)
        _, file = file_obj
        if hasattr(file, "read"):
            pos = file.tell()
            file.seek(0)
            content = file.read()
            file.seek(pos)
            return content
        elif os.path.isfile(file):
            with open(file, "rb") as f:
                return f.read()
    
    raise ValueError(f"Unsupported file type: {type(file_obj)}")

def save_images_to_cache(file_hash, images, filename=None):
    """Save converted images to cache directory."""
    try:
        cache_path = os.path.join(CACHE_DIR, f"{file_hash}.cache")
        
        # Serialize metadata
        metadata = {
            "count": len(images),
            "filename": filename,
            "timestamp": os.path.getmtime(cache_path) if os.path.exists(cache_path) else None
        }
        
        # Save metadata
        with open(os.path.join(CACHE_DIR, f"{file_hash}.meta"), "wb") as f:
            pickle.dump(metadata, f)
        
        # Save individual images as files
        for i, img in enumerate(images):
            img_path = os.path.join(CACHE_DIR, f"{file_hash}_{i}.jpg")
            img.save(img_path, "JPEG")
        
        logger.info(f"Cached {len(images)} images for file {filename or file_hash}")
        return True
    except Exception as e:
        logger.error(f"Failed to cache images: {e}")
        return False

def load_images_from_cache(file_hash):
    """Load images from cache if they exist."""
    try:
        # Check if metadata exists
        meta_path = os.path.join(CACHE_DIR, f"{file_hash}.meta")
        if not os.path.exists(meta_path):
            return None
        
        # Load metadata
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
        
        # Load images
        images = []
        for i in range(metadata["count"]):
            img_path = os.path.join(CACHE_DIR, f"{file_hash}_{i}.jpg")
            if os.path.exists(img_path):
                images.append(Image.open(img_path))
            else:
                logger.warning(f"Missing cached image {i} for {file_hash}")
                return None  # If any image is missing, regenerate all
        
        logger.info(f"Loaded {len(images)} cached images for {metadata.get('filename', file_hash)}")
        return images
    except Exception as e:
        logger.error(f"Failed to load cached images: {e}")
        return None

def convert_files(files, page_limit=1000):
    """
    Convert uploaded files to images, with caching for efficiency.
    
    Args:
        files: List of file objects from Gradio
        page_limit: Maximum number of pages to process
        
    Returns:
        List of PIL Image objects
    """
    images = []
    processed_files = 0
    
    for f in files:
        try:
            # Extract filename for logging
            if hasattr(f, "name"):
                filename = os.path.basename(f.name)
            elif isinstance(f, tuple) and len(f) == 2:
                filename = os.path.basename(f[0])
            elif isinstance(f, str):
                filename = os.path.basename(f)
            else:
                filename = f"unknown_file_{processed_files}"
            
            logger.info(f"Processing file: {filename}")
            
            try:
                # Get file content for hashing
                file_content = get_file_content(f)
                file_hash = get_file_hash(file_content)
                
                # Check if we have this file cached
                cached_images = load_images_from_cache(file_hash)
                if cached_images:
                    logger.info(f"Using cached images for {filename}")
                    images.extend(cached_images)
                    processed_files += 1
                    continue
                
                # If not cached, convert the file
                logger.info(f"No cache found for {filename}, converting...")
                
                # Process the file based on its type
                if hasattr(f, "name"):
                    # File object with name attribute
                    file_path = f.name
                    new_images = convert_from_path(file_path, thread_count=4)
                elif isinstance(f, tuple) and len(f) == 2:
                    # Tuple of (name, file-like object)
                    temp_name, temp_file = f
                    if hasattr(temp_file, "read"):
                        # File-like object, read and convert from bytes
                        file_content = temp_file.read() if file_content is None else file_content
                        new_images = convert_from_bytes(file_content, thread_count=4)
                    else:
                        # Path string
                        new_images = convert_from_path(temp_file, thread_count=4)
                elif isinstance(f, str):
                    # Direct file path
                    new_images = convert_from_path(f, thread_count=4)
                else:
                    # Try to handle other types
                    if hasattr(f, "file"):
                        file_content = f.file.read() if file_content is None else file_content
                        new_images = convert_from_bytes(file_content, thread_count=4)
                    elif hasattr(f, "read"):
                        file_content = f.read() if file_content is None else file_content
                        new_images = convert_from_bytes(file_content, thread_count=4)
                    else:
                        raise TypeError(f"Unsupported file type: {type(f)}")
                
                # Cache the newly converted images
                if new_images:
                    save_images_to_cache(file_hash, new_images, filename)
                    images.extend(new_images)
                    processed_files += 1
                
            except Exception as e:
                logger.error(f"Error converting file {filename}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        except Exception as e:
            logger.error(f"Error processing file entry: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Check total page count
    if len(images) > page_limit:
        logger.warning(f"Exceeding page limit ({len(images)} > {page_limit}), truncating")
        images = images[:page_limit]
        raise ValueError(f"The number of images in the dataset exceeds the limit of {page_limit}. Only the first {page_limit} pages will be processed.")
    
    if not images:
        logger.error("No valid images were generated from the provided files")
        raise ValueError("No valid PDF files were processed. Please check your uploads.")
    
    logger.info(f"Successfully processed {processed_files} files with {len(images)} total pages")
    return images