import os
import spaces
import base64
from io import BytesIO
import io
import requests

import gradio as gr
import torch

from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from colpali_engine.models import ColQwen2, ColQwen2Processor

import sys

from db import DocumentEmbeddingDatabase

import zipfile
from datetime import datetime
from dotenv import load_dotenv
from image_cache import convert_files

load_dotenv()

# Create the directory for the embeddings database if it doesn't exist
os.makedirs("./data/embeddings_db", exist_ok=True)

# Initialize the database - this connects to the local file-based database
db = DocumentEmbeddingDatabase(db_path="./data/embeddings_db")

print(f"Python path: {sys.executable}")
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch path: {torch.__file__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS (Apple Silicon) available: {torch.backends.mps.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
elif torch.backends.mps.is_available():
    print("Running on Apple Silicon GPU")

# Define model paths
MODEL_DIR = "./models/colqwen2"
MODEL_PATH = os.path.join(MODEL_DIR, "model")
PROCESSOR_PATH = os.path.join(MODEL_DIR, "processor")
MODEL_MARKER = os.path.join(MODEL_DIR, "model_loaded.marker")

PAGE_LIMIT = 1000

# LLM Provider options for the dropdown
LLM_PROVIDERS = ["OpenAI GPT-4o-mini", "Anthropic Claude 3.7 Sonnet", "Ollama llama3.2-vision:11b"]

def get_api_key(llm_provider):
    index = LLM_PROVIDERS.index(llm_provider)
    if index == 0:
        return os.getenv("OPENAI_API_KEY","")
    if index == 1:
        return os.getenv("ANTHROPIC_API_KEY","")
    if index == 2:
        return ""

def verify_model_directories():
    """Verify that model directories exist and are writable."""
    print("=== Verifying Model Directories ===")

    # Ensure directories exist
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(PROCESSOR_PATH, exist_ok=True)

    print(f"MODEL_DIR: {MODEL_DIR} - Exists: {os.path.exists(MODEL_DIR)}")
    print(f"MODEL_PATH: {MODEL_PATH} - Exists: {os.path.exists(MODEL_PATH)}")
    print(
        f"PROCESSOR_PATH: {PROCESSOR_PATH} - Exists: {os.path.exists(PROCESSOR_PATH)}"
    )

    # Verify we can write to these directories
    try:
        test_file_model = os.path.join(MODEL_PATH, "test_write.tmp")
        test_file_processor = os.path.join(PROCESSOR_PATH, "test_write.tmp")

        with open(test_file_model, "w") as f:
            f.write("test")
        with open(test_file_processor, "w") as f:
            f.write("test")

        # Clean up
        os.remove(test_file_model)
        os.remove(test_file_processor)
        print(
            "✅ Model directories are writable - volume mounting is working correctly"
        )
        return True
    except Exception as e:
        print(
            f"❌ Error: Cannot write to model directories. Docker volume may not be mounted correctly: {e}"
        )
        print(
            "Please ensure that './models' directory exists and has proper permissions"
        )
        return False


def check_model_persistence():
    """Check if model persistence marker exists from previous runs."""
    if os.path.exists(MODEL_MARKER):
        with open(MODEL_MARKER, "r") as f:
            marker_content = f.read()
            print(f"✅ Model persistence confirmed! Previous marker: {marker_content}")
        return True
    print("No model persistence marker found - this might be the first run")
    return False


def mark_model_loaded():
    """Create a marker file indicating the model has been loaded successfully."""
    try:
        with open(MODEL_MARKER, "w") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"Model loaded successfully at {timestamp}")
        print(f"Created model marker file at {MODEL_MARKER}")
    except Exception as e:
        print(f"Warning: Could not create model marker file: {e}")


def create_zip_for_download(query, response, images):
    """
    Create a zip file containing the query, response, and retrieved images.

    Args:
        query (str): The user's query
        response (str): The AI's response to the query
        images (list): List of (image, caption) tuples from the search results

    Returns:
        bytes: The zip file as bytes
    """
    # Create an in-memory zip file
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Create markdown file with query and response
        markdown_content = f"# Query\n\n{query}\n\n# AI Response\n\n{response}"
        zip_file.writestr("query_response.md", markdown_content)

        # Add images to zip file
        for i, (image, caption) in enumerate(images):
            # Save image to bytes
            img_buffer = io.BytesIO()
            image.save(img_buffer, format="JPEG")
            img_buffer.seek(0)

            # Add to zip with caption as part of filename
            clean_caption = caption.replace("/", "_").replace("\\", "_")
            zip_file.writestr(
                f"image_{i + 1}_{clean_caption}.jpg", img_buffer.getvalue()
            )

    # Return the zip file as bytes
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


@spaces.GPU
def install_fa2():
    print("Install FA2")
    os.system("pip install flash-attn --no-build-isolation")


# install_fa2()  # Disabled for Docker


def load_model():
    """Load model from disk if available, otherwise download and save it."""
    try:
        # Verify directories are properly set up before proceeding
        if not verify_model_directories():
            print(
                "WARNING: Model directories verification failed, but will try to continue"
            )

        # Check if model was previously loaded successfully
        model_persistence = check_model_persistence()

        os.makedirs(MODEL_DIR, exist_ok=True)
        print(f"Model directory: {MODEL_DIR}")
        print(f"Model directory exists: {os.path.exists(MODEL_DIR)}")

        # Check for Apple Silicon and set device appropriately
        if torch.backends.mps.is_available():
            device = "mps"
            print("Apple Silicon detected, using MPS device")
        elif torch.cuda.is_available():
            device = "cuda:0"
            print("NVIDIA GPU detected, using CUDA device")
        else:
            device = "cpu"
            print("No GPU detected, using CPU (slow)")

        print(f"Using device: {device}")

        # Print additional hardware info based on device type
        if device == "cuda:0":
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        elif device == "mps":
            print("Running on Apple Silicon")

        # Check if critical model files exist - any of these combinations are valid
        model_files_exist = (
            os.path.exists(os.path.join(MODEL_PATH, "config.json"))
        ) or (
            os.path.exists(os.path.join(MODEL_PATH, "adapter_config.json"))
            and os.path.exists(os.path.join(MODEL_PATH, "adapter_model.safetensors"))
            and os.path.exists(os.path.join(MODEL_PATH, "generation_config.json"))
        )

        processor_files_exist = (
            os.path.exists(os.path.join(PROCESSOR_PATH, "config.json"))
        ) or (
            os.path.exists(os.path.join(PROCESSOR_PATH, "tokenizer_config.json"))
            and os.path.exists(os.path.join(PROCESSOR_PATH, "tokenizer.json"))
            and os.path.exists(os.path.join(PROCESSOR_PATH, "vocab.json"))
        )

        print(f"Model files exist: {model_files_exist}")
        print(f"Processor files exist: {processor_files_exist}")

        # List files in model directory to debug
        if os.path.exists(MODEL_PATH):
            print(f"Files in model directory: {os.listdir(MODEL_PATH)}")
        if os.path.exists(PROCESSOR_PATH):
            print(f"Files in processor directory: {os.listdir(PROCESSOR_PATH)}")

        # Only attempt to load if critical files exist
        if model_files_exist and processor_files_exist:
            print("Loading model from disk - step 1...")
            try:
                # Use absolute paths to avoid any reference issues
                abs_model_path = os.path.abspath(MODEL_PATH)
                abs_processor_path = os.path.abspath(PROCESSOR_PATH)
                print(f"Using absolute model path: {abs_model_path}")

                # Load model with trust_remote_code
                print("Loading model from disk - step 2...")
                model = ColQwen2.from_pretrained(
                    abs_model_path,
                    torch_dtype=torch.bfloat16,
                    device_map=device,
                    local_files_only=True,  # Changed to True to force local loading
                    trust_remote_code=True,
                    revision=None,  # Important: don't try to fetch remote info
                )
                print("Model loaded successfully!")

                print("Loading processor...")
                processor = ColQwen2Processor.from_pretrained(
                    abs_processor_path,
                    local_files_only=True,
                    trust_remote_code=True,
                    revision=None,
                )
                print("Processor loaded successfully!")

                print("Putting model in evaluation mode...")
                model = model.eval()
                print("Model ready!")

                # Mark that model was loaded successfully
                mark_model_loaded()

                return model, processor
            except Exception as e:
                print(f"Error loading model from disk: {e}")
                print(f"Error type: {type(e)}")
                print("Forcing download of a new model...")
                # Force download new model
                return download_model(device)
        else:
            print("Model files not found or incomplete on disk, downloading...")
            return download_model(device)
    except Exception as e:
        print(f"Exception in load_model: {e}")
        print(f"Exception type: {type(e)}")
        raise


def download_model(device):
    print("Downloading model (first run only)...")
    try:
        print("Download step 1 - initializing...")
        model = ColQwen2.from_pretrained(
            "vidore/colqwen2-v1.0",
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        print("Download step 2 - model downloaded successfully!")

        print("Setting model to eval mode...")
        model = model.eval()

        print("Downloading processor...")
        processor = ColQwen2Processor.from_pretrained(
            "vidore/colqwen2-v1.0", trust_remote_code=True
        )
        print("Processor downloaded successfully!")

        # Save model and processor to disk with absolute paths
        print("Saving model to disk for future use...")
        try:
            abs_model_path = os.path.abspath(MODEL_PATH)
            abs_processor_path = os.path.abspath(PROCESSOR_PATH)

            print(f"Saving model to {abs_model_path}...")
            model.save_pretrained(abs_model_path)
            print(f"Model saved successfully!")

            print(f"Saving processor to {abs_processor_path}...")
            processor.save_pretrained(abs_processor_path)
            print(f"Processor saved successfully!")

            # Mark that model was loaded and saved successfully
            mark_model_loaded()

        except Exception as e:
            print(f"Error saving model to disk: {e}")
            print(f"Error type: {type(e)}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        print(f"Error type: {type(e)}")
        raise

    return model, processor


# Verify model directories and persistence before loading
verify_model_directories()
check_model_persistence()

# Load model and processor
model, processor = load_model()


def encode_image_to_base64(image):
    """Encodes a PIL image to a base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

LLM_PROVIDERS = ["OpenAI GPT-4o-mini", "Anthropic Claude 3.7 Sonnet", "Ollama llama3.2-vision:11b"]

def query_ollama(query, images, api_key=None):
    """
    Calls Ollama with the llama3.2-vision:11b model using direct HTTP requests,
    processes one image at a time, and joins all responses into one comprehensive answer.
    """
    import requests
    import json
    import tempfile
    import os
    import base64
    
    all_responses = []
    
    for i, (img, caption) in enumerate(images):
        try:
            # Create a temporary directory for this single image
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, f"image_{i}.jpg")
            img.save(temp_path, "JPEG")
            
            # Format the prompt
            prompt = f"""You are a smart assistant designed to answer questions about a PDF document.
You are given relevant information in the form of PDF pages. Use them to construct a detailed response to the question, and cite your sources (page numbers, etc).
If it is not possible to answer using the provided pages, do not attempt to provide an answer and simply say the answer is not present within the documents.
Give detailed and extensive answers, only containing info in the pages you are given.
You can answer using information contained in plots and figures if necessary.
Answer in the same language as the query.
Query: {query}
PDF page {i+1}: (see attached image)
"""
            
            # Read and encode the single image
            with open(temp_path, "rb") as img_file:
                encoded = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Create the payload for the API request with just one image
            payload = {
                "model": "llama3.2-vision:11b",
                "prompt": prompt,
                "images": [encoded],  # Just one image in the list
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 2048
                }
            }
            
            # Send the request to Ollama
            response = requests.post(
                "http://localhost:11434/api/generate",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=120
            )
            
            # Clean up temporary files
            try:
                os.remove(temp_path)
                os.rmdir(temp_dir)
            except:
                pass
            
            # Check if the request was successful
            if response.status_code == 200:
                result = response.json()
                all_responses.append(f"--- Response for Page {i+1} ---\n{result.get('response', f'No content for page {i+1}')}")
            else:
                all_responses.append(f"Error for page {i+1}: Ollama returned status code {response.status_code}. Response: {response.text}")
            
        except Exception as e:
            import traceback
            error_msg = f"Ollama API connection failure for page {i+1}. Error: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)  # Log the error for debugging
            all_responses.append(error_msg)
    
    # Join all responses into one comprehensive answer
    combined_response = "\n\n".join(all_responses)
    
    # Add a summary header
    final_response = f"Combined Analysis for Query: {query}\n\n{combined_response}"
    
    return final_response
def query_claude(query, images, api_key):
    """Calls Anthropic's Claude 3.7 Sonnet with the query and image data."""

    if not api_key or not api_key.startswith("sk-ant"):
        return "Enter your Anthropic API key to get a response from Claude 3.7 Sonnet"

    try:
        # Format the Claude prompt
        CLAUDE_PROMPT = """
        You are a smart assistant designed to answer questions about a PDF document.
        You are given relevant information in the form of PDF pages. Use them to construct a detailed response to the question, and cite your sources (page numbers, etc).
        If it is not possible to answer using the provided pages, do not attempt to provide an answer and simply say the answer is not present within the documents.
        Give detailed and extensive answers, only containing info in the pages you are given.
        You can answer using information contained in plots and figures if necessary.
        Answer in the same language as the query.
        
        Query: {query}
        PDF pages:
        """

        # Create the Anthropic API request
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key.strip(),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        # Prepare the message content with text and images
        content = [{"type": "text", "text": CLAUDE_PROMPT.format(query=query)}]

        # Add images to the content
        for i, (image, caption) in enumerate(images):
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_bytes = buffered.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")

            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img_base64,
                    },
                }
            )

            # Add caption as text after each image
            content.append({"type": "text", "text": f"\n{caption}\n"})

        # Construct the final request payload
        data = {
            "model": "claude-3-7-sonnet-20250219",
            "max_tokens": 4000,
            "messages": [{"role": "user", "content": content}],
        }

        # Send the request to Anthropic
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()

        return result["content"][0]["text"]

    except Exception as e:
        return f"Anthropic API connection failure. Error: {e}"


def query_gpt4o_mini(query, images, api_key):
    """Calls OpenAI's GPT-4o-mini with the query and image data."""

    if not api_key or not api_key.startswith("sk-"):
        return "Enter your OpenAI API key to get a response from GPT-4o-mini"

    try:
        from openai import OpenAI

        base64_images = [encode_image_to_base64(image[0]) for image in images]
        client = OpenAI(api_key=api_key.strip())
        PROMPT = """
        You are a smart assistant designed to answer questions about a PDF document.
        You are given relevant information in the form of PDF pages. Use them to construct a detailed response to the question, and cite your sources (page numbers, etc).
        If it is not possible to answer using the provided pages, do not attempt to provide an answer and simply say the answer is not present within the documents.
        Give detailed and extensive answers, only containing info in the pages you are given.
        You can answer using information contained in plots and figures if necessary.
        Answer in the same language as the query.
        
        Query: {query}
        PDF pages:
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": PROMPT.format(query=query)}]
                    + [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{im}"},
                        }
                        for im in base64_images
                    ],
                }
            ],
            max_tokens=8000,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenAI API connection failure. Error: {e}"


def parse_api_keys(api_key_input):
    """
    Parse API key input that might contain multiple keys in format:
    openai:sk-xxx|anthropic:sk-ant-xxx or just a single key

    Returns a dictionary of provider:key pairs
    """
    api_keys = {}

    if "|" in api_key_input:
        for key_pair in api_key_input.split("|"):
            if ":" in key_pair:
                provider, key = key_pair.split(":", 1)
                api_keys[provider.strip().lower()] = key.strip()
    else:
        # Try to determine the provider based on key format
        key = api_key_input.strip()
        if key.startswith("sk-ant"):
            api_keys["anthropic"] = key
        elif key.startswith("sk-"):
            api_keys["openai"] = key

    return api_keys


def query_llm(query, images, api_key_input, llm_provider):
    """
    Route the query to the appropriate LLM based on the selected provider.

    Args:
        query (str): The user's query
        images (list): List of (image, caption) tuples from the search results
        api_key_input (str): The API key for the selected provider
        llm_provider (str): The provider/model to use

    Returns:
        str: The LLM's response
    """
    # Parse API keys
    api_keys = parse_api_keys(api_key_input)

    # Handle the routing based on provider
    if llm_provider == "OpenAI GPT-4o-mini":
        # Check if we have a specific OpenAI key
        openai_key = api_keys.get("openai", api_key_input)
        return query_gpt4o_mini(query, images, openai_key)

    elif llm_provider == "Anthropic Claude 3.7 Sonnet":
        # Check if we have a specific Anthropic key
        anthropic_key = api_keys.get("anthropic", api_key_input)
        return query_claude(query, images, anthropic_key)
    elif llm_provider == "Ollama llama3.2-vision:11b":
        # Ollama doesn't need API key as it's running locally
        return query_ollama(query, images)
    else:
        return f"Unknown LLM provider: {llm_provider}. Please select a valid option."


# Modify the search function to correctly format the return value for Gradio File component
@spaces.GPU
def search(query: str, ds, images, k, api_key, llm_provider):
    try:
        k = min(k, len(ds))
        # Check for Apple Silicon and set device appropriately
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"

        if device != model.device:
            model.to(device)

        qs = []
        with torch.no_grad():
            batch_query = processor.process_queries([query]).to(model.device)
            embeddings_query = model(**batch_query)
            qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

        scores = processor.score(qs, ds, device=device)

        top_k_indices = scores[0].topk(k).indices.tolist()

        results = []
        for idx in top_k_indices:
            results.append((images[idx], f"Page {idx}"))

        # Generate response from the selected LLM
        ai_response = query_llm(query, results, api_key, llm_provider)

        # Create download data
        download_data = None
        if results and ai_response:
            try:
                # Create timestamp for download filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"colpali_results_{timestamp}.zip"

                # Create the temporary file path where we'll save the zip
                import tempfile

                temp_dir = os.path.join(tempfile.gettempdir(), "colpali_downloads")
                os.makedirs(temp_dir, exist_ok=True)
                temp_zip_path = os.path.join(temp_dir, filename)

                # Create zip file content
                zip_data = create_zip_for_download(query, ai_response, results)

                # Write to temp file
                with open(temp_zip_path, "wb") as f:
                    f.write(zip_data)

                # Return the file path for Gradio File component
                download_data = temp_zip_path
            except Exception as e:
                print(f"Error creating download file: {e}")
                import traceback

                traceback.print_exc()

        return results, ai_response, download_data
    except Exception as e:
        import traceback

        error_msg = f"Error in search function: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return [], error_msg, None


# def convert_files(files):
#     """Convert uploaded files to images, handling different file types from Gradio."""
#     images = []

#     for f in files:
#         try:
#             # Handle the file based on its type
#             if hasattr(f, "name"):
#                 # This is likely a file object with a name attribute
#                 file_path = f.name
#                 print(f"Processing file with path: {file_path}")
#                 images.extend(convert_from_path(file_path, thread_count=4))
#             elif isinstance(f, tuple) and len(f) == 2:
#                 # If it's a tuple of (name, file-like object) as returned by some Gradio versions
#                 temp_name, temp_file = f
#                 print(f"Processing tuple with name: {temp_name}")
#                 # If it's a file-like object, read it and convert from bytes
#                 if hasattr(temp_file, "read"):
#                     file_content = temp_file.read()
#                     images.extend(convert_from_bytes(file_content, thread_count=4))
#                 else:
#                     # If it's a path
#                     images.extend(convert_from_path(temp_file, thread_count=4))
#             elif isinstance(f, str):
#                 # If it's directly a file path
#                 print(f"Processing file path: {f}")
#                 images.extend(convert_from_path(f, thread_count=4))
#             else:
#                 # Try to get the file path from the object
#                 print(f"Unknown file type: {type(f)}, trying to handle generically")
#                 if hasattr(f, "file"):
#                     # Some Gradio versions provide a file attribute
#                     file_content = f.file.read()
#                     images.extend(convert_from_bytes(file_content, thread_count=4))
#                 elif hasattr(f, "read"):
#                     # If it's a file-like object
#                     file_content = f.read()
#                     images.extend(convert_from_bytes(file_content, thread_count=4))
#                 else:
#                     raise TypeError(
#                         f"Unsupported file type: {type(f)}. Please provide a valid PDF file."
#                     )
#         except Exception as e:
#             print(f"Error processing file {f}: {e}")
#             import traceback

#             traceback.print_exc()
#             # Continue with other files rather than failing completely
#             continue

#     if len(images) > PAGE_LIMIT:
#         raise gr.Error(
#             f"The number of images in the dataset should be less than {PAGE_LIMIT}."
#         )

#     if not images:
#         raise ValueError(
#             "No valid PDF files were processed. Please check your uploads."
#         )

#     return images


def index(files, ds):
    try:
        print("Converting files")
        print(f"File types: {[type(f) for f in files]}")

        # Reset the embeddings list and images list
        ds = []
        all_images = []

        # Use the enhanced convert_files function that implements caching
        try:
            # This will use cached images when available
            all_images = convert_files(files)
            print(f"Successfully converted {len(all_images)} pages from {len(files)} files")
        except Exception as e:
            print(f"Error converting files: {e}")
            import traceback
            traceback.print_exc()
            return f"Error converting files: {str(e)}", ds, []

        # Process embeddings for each file
        for f in files:
            # Get filename for identification
            if hasattr(f, "name"):
                file_path = f.name
            elif isinstance(f, tuple) and len(f) == 2:
                file_path = f[0]
            elif isinstance(f, str):
                file_path = f
            else:
                if hasattr(f, "file"):
                    file_path = str(f.file)
                else:
                    import hashlib
                    file_path = f"unknown_file_{hashlib.md5(str(f).encode()).hexdigest()}"

            filename = os.path.basename(file_path)
            
            # Check if embeddings exist for this file
            if db.embeddings_exist(filename):
                print(f"Loading existing embeddings for {filename}")
                
                # Load embeddings from database using the filename
                file_embeddings = db.load_embeddings(filename)
                
                if file_embeddings and len(file_embeddings) > 0:
                    # Add to our embeddings list
                    ds.extend(file_embeddings)
                    print(f"Loaded {len(file_embeddings)} existing embeddings")
                else:
                    print(f"Failed to load embeddings, will regenerate")
                    # Fall back to generating new embeddings
                    process_new_file(f, filename, ds, all_images)
            else:
                print(f"No existing embeddings for {filename}, generating new ones")
                # Process the file normally
                process_new_file(f, filename, ds, all_images)

        return f"Processed {len(files)} files with {len(ds)} total embeddings", ds, all_images
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        return f"Error in indexing: {str(e)}\n{traceback_str}", ds, []
    
def process_new_file(f, file_id, ds, all_images):
    """Process a file by generating new embeddings and saving them"""
    try:
        # Find the relevant images for this file
        # Note: This assumes all_images has been populated with all pages from all files
        # We need to find only the images for this specific file
        
        # If we don't already have images for this file, convert it
        if not all_images:
            print(f"Converting file {file_id} to images")
            file_images = convert_files([f])
        else:
            # All images are already loaded, so we're good to proceed
            file_images = all_images
            
        # Get embeddings for this file
        file_ds = []
        status, file_ds, _ = index_gpu(file_images, file_ds)
        
        # Save the new embeddings if successful
        if file_ds and len(file_ds) > 0:
            saved = db.save_embeddings(file_id, file_ds, len(file_images))
            if saved:
                print(f"Saved {len(file_ds)} new embeddings for {file_id}")
            else:
                print(f"Failed to save embeddings for {file_id}")
                
            # Add to our complete embeddings list
            ds.extend(file_ds)
        else:
            print(f"Failed to generate embeddings for {file_id}")
    except Exception as e:
        print(f"Error processing new file {file_id}: {e}")
        import traceback
        traceback.print_exc()
    
# Modified index_gpu function (keeping the core functionality the same)
@spaces.GPU
def index_gpu(images, ds):
    """Example script to run inference with ColPali (ColQwen2)"""
    try:
        # Check for Apple Silicon and set device appropriately
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"

        if device != model.device:
            model.to(device)

        # run inference - docs
        dataloader = DataLoader(
            images,
            batch_size=1,
            # num_workers=2,
            shuffle=False,
            collate_fn=lambda x: processor.process_images(x).to(model.device),
        )

        for batch_doc in tqdm(dataloader):
            with torch.no_grad():
                batch_doc = {k: v.to(device) for k, v in batch_doc.items()}
                embeddings_doc = model(**batch_doc)
            ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))
        return f"Uploaded and converted {len(images)} pages", ds, images
    except Exception as e:
        import traceback

        traceback_str = traceback.format_exc()
        return f"Error in processing: {str(e)}\n{traceback_str}", ds, []


# Update the Gradio UI section for better file handling
with gr.Blocks(theme=gr.themes.Glass(), title="RTFM") as demo:
    gr.Markdown("# ®️etrieval For Technical Ⓜ️anuals (RTFM) 😜")
    with gr.Accordion("Details:", open=False):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(""" ## Problem: ## 
                            Typical RAG fails for highly graphical documents.
                            1. Can only parse text
                            2. Poor understanding of graphical elements
                            3. Typical AI chatbot at most can accept a few images or small documents. Unable to process manuals with hundreds of pages.
                            4. No document persistence.
                            """)
            with gr.Column(scale=1):
                gr.Markdown(""" ## Solution ##
                    The ColPali model's results are promising! 
                    1. ColPali's strategy is to ingest each pdf page as an image. These embeddings are stored in a LanceDB embeddings database for persistence. The query is first sent to the Colpali model, returning responses in the form of images with page number.
                    2. For the best result send the query to cloud AI (OpenAI GPT-4o-mini or Anthropic Claude 3.7) to get the final response.
                    3. For privacy, send use local model (llama3.2-vision:11b) to get the final response.
                    4. Save the result as a zip file for easy reference.
                    """)

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## 1️⃣ Upload PDFs")
            file = gr.File(file_count="multiple", label="Upload PDFs")

            convert_button = gr.Button("🔄 Index documents")
            message = gr.Textbox("Files not yet uploaded", label="Status")

            # Create a dropdown for the LLM provider
            llm_provider = gr.Dropdown(
                choices=LLM_PROVIDERS,
                label="Select AI Model for Response Generation",
                value=LLM_PROVIDERS[0],
                info="Choose which AI model will answer your questions using the retrieved PDF pages.",
            )

            # Add API key input with improved description
            api_key = gr.Textbox(
                placeholder="Enter API key(s): openai:sk-xxx|anthropic:sk-ant-xxx or just paste single key",
                label="API key",
                value=get_api_key(llm_provider.value),
                type="password",
                info="Enter your OpenAI or Anthropic API key",
            )

            embeds = gr.State(value=[])
            imgs = gr.State(value=[])

        with gr.Column(scale=3):
            gr.Markdown("## 2️⃣ Search")
            query = gr.Textbox(placeholder="Enter your query here", label="Query")
            k = gr.Slider(
                minimum=1, maximum=10, step=1, label="Number of results", value=5
            )
            search_button = gr.Button("🔍 Search", variant="primary")

    # Define the outputs first
    output_gallery = gr.Gallery(
        label="Retrieved Documents",
        height=800,
        show_label=True,
        show_share_button=True,
        columns=[5],
        rows=[1],
        object_fit="contain",
    )
    output_text = gr.Textbox(
        label="AI Response",
        placeholder="Generated response based on retrieved documents",
        show_copy_button=True,
    )
    download_file = gr.File(label="Download Results", visible=False)

    # Define the actions
    convert_button.click(index, inputs=[file, embeds], outputs=[message, embeds, imgs])

    # Update search button click to include the LLM provider
    search_result = search_button.click(
        search,
        inputs=[query, embeds, imgs, k, api_key, llm_provider],
        outputs=[output_gallery, output_text, download_file],
    )

    # Add event listener to update API key when LLM provider changes
    llm_provider.change(
        fn=lambda provider: os.getenv("OPENAI_API_KEY", "") 
        if provider == LLM_PROVIDERS[0]
        else "" if provider == "Ollama llama3.2-vision:11b"
        else os.getenv("ANTHROPIC_API_KEY", ""),
        inputs=[llm_provider],
        outputs=[api_key],
    )

    # Show download file when search completes with results
    search_result.then(
        lambda file: gr.update(visible=file is not None), [download_file], download_file
    )

    with gr.Accordion("API Keys Usage:", open=False):
        gr.Markdown("""
            ## API Key Format
            - For OpenAI GPT-4o-mini: Enter your OpenAI API key starting with `sk-`
            - For Anthropic Claude 3.7: Enter your Anthropic API key starting with `sk-ant-`
            - For Ollama llama3.2-vision:11b: No API key needed (uses localhost:11434)
            
            ## Combined Format (Optional)
            You can provide both API keys in the format: `openai:sk-xxx|anthropic:sk-ant-xxx`
            
            This allows you to switch between models without changing the API key each time.
            """)

    with gr.Accordion("Acknowledgements:", open=False):
        gr.Markdown(
            "# ColPali: Efficient Document Retrieval with Vision Language Models (ColQwen2) 📚"
        )
        gr.Markdown("""Demo to test ColQwen2 (ColPali) on PDF documents. 
        ColPali is model implemented from the [ColPali paper](https://arxiv.org/abs/2407.01449).

        This demo allows you to upload PDF files and search for the most relevant pages based on your query.
        Refresh the page if you change documents!

        ⚠️ This demo uses a model trained exclusively on A4 PDFs in portrait mode, containing english text. Performance is expected to drop for other page formats and languages.
        Other models will be released with better robustness towards different languages and document formats!
        """)

if __name__ == "__main__":
    # Use a simpler launch method to avoid compatibility issues
    import atexit
    import shutil
    import tempfile

    # Create a temporary directory for file uploads if it doesn't exist
    temp_dir = os.path.join(tempfile.gettempdir(), "colpali_uploads")
    os.makedirs(temp_dir, exist_ok=True)

    # Clean up function to remove temp files
    def cleanup():
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    # Register the cleanup function
    atexit.register(cleanup)

    # Launch Gradio with simplified server settings
    demo.queue(max_size=10).launch(
        server_name="0.0.0.0", server_port=7860, share=False, debug=True
    )
