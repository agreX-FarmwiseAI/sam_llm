# --- START OF FILE new_sam.py ---

import os
import json
import time
from tqdm import tqdm
import google.generativeai as genai
from datetime import datetime, timedelta
import pandas as pd # Added for potential future use, though not strictly needed for current logic
import geopandas as gpd
from rasterio.mask import mask
import rasterio
from rasterio.crs import CRS # Import CRS object
from PIL import Image
import numpy as np
from samgeo import SamGeo, tms_to_geotiff

# --- LLM Configuration (Merged from old_ref_llm.py) ---
# WARNING: Hardcoding API keys is insecure. Consider using environment variables or a secrets manager.
API_KEYS = [
    "AIzaSyCNb64e_usdhwZd8GLzvngHizyK1wERJ2Q", # Replace with your actual keys
    "AIzaSyA9sRX75ZhHWdPG1UAPD6nCPIN7K6g9Vm4", # Replace with your actual keys
    "AIzaSyC9c3jdozjLBUIrVK0UPPuo9UD7Y3IB2aU", # Replace with your actual keys
    "AIzaSyDxVY2daczzAbkQOZVMcB5HiRY_2xTl02w", # Replace with your actual keys
    "AIzaSyAc2MJMkPQGoP9P26oFw3wxtpU-CrFMdqA", # Replace with your actual keys
    "AIzaSyAeRLHOyj_fwpaLA7fEc2KsvrII9TfcWKE", # Replace with your actual keys
    "AIzaSyBU3k3MFcWr9Xd2eyrAKZK-aT2OIX9PstU", # Replace with your actual keys
]
MODEL_NAME = "gemini-2.0-flash" # Use a known valid model like 1.5 flash
MIN_KEY_COOLDOWN = 8  # Minimum seconds between uses of the same API key

# Track when each key was last used
last_used = {key: datetime.now() - timedelta(seconds=MIN_KEY_COOLDOWN * 2) for key in API_KEYS} # Initialize to allow immediate use

# Load the LLM Prompt
try:
    with open("sam_prompt.txt", "r") as f:
        llm_prompt = f.read()
except FileNotFoundError:
    print("Error: sam_prompt.txt not found. Please ensure the prompt file exists.")
    llm_prompt = None # Set to None to handle the error later

generation_config = {
    "temperature": 0.5,
    "top_p": 0.5,
    "top_k": 64,
    "max_output_tokens": 8182, # Adjust if needed for the prompt/model
    "response_mime_type": "application/json", # Request JSON directly
}

# --- LLM Helper Functions (Merged from old_ref_llm.py) ---

def get_next_available_key():
    """Gets the next available API key respecting the cooldown period."""
    if not API_KEYS:
        raise ValueError("No API keys provided.")

    while True: # Loop until a key is found or wait time is calculated
        now = datetime.now()
        # Sort keys by when they were last used, oldest first
        sorted_keys = sorted(last_used.items(), key=lambda x: x[1])

        for key, last_use_time in sorted_keys:
            time_since_last_use = (now - last_use_time).total_seconds()
            if time_since_last_use >= MIN_KEY_COOLDOWN:
                last_used[key] = now  # Update the last used time
                # print(f"Using API key ending with ...{key[-5:]}") # Optional: for debugging
                return key

        # If all keys are on cooldown, wait for the one that will be available soonest
        soonest_key, soonest_time = sorted_keys[0]
        wait_time = MIN_KEY_COOLDOWN - (now - soonest_time).total_seconds()
        if wait_time > 0:
            print(f"All API keys on cooldown, waiting {wait_time:.2f} seconds...")
            time.sleep(wait_time + 0.1) # Add a small buffer
        # Loop again to re-check availability after waiting

def call_llm(image_path):
    """Calls the Gemini LLM to classify the image segment using the next available API key."""
    if not llm_prompt:
        print("Error: LLM prompt is not loaded. Cannot perform classification.")
        return None
    if not API_KEYS:
        print("Error: No API keys configured. Cannot perform classification.")
        return None

    api_key = get_next_available_key()

    try:
        # Configure genai with the selected API key
        genai.configure(api_key=api_key)
        # Ensure model name is correct and available
        model = genai.GenerativeModel(MODEL_NAME) # Removed generation_config here, applied in generate_content

        print(f"Classifying {os.path.basename(image_path)} using API key ...{api_key[-5:]}")

        # Prepare image part
        image_part = {
            "mime_type": "image/png", # We are using PNG files
            "data": open(image_path, "rb").read()
        }

        # Prepare prompt parts
        prompt_parts = [llm_prompt, "INPUT IMAGE:", image_part]

        # Make the API call
        response = model.generate_content(prompt_parts, generation_config=generation_config)

        # Debug: Print raw response text
        # print("Raw LLM Response:", response.text)

        if response.text:
            try:
                # The response should be directly JSON if response_mime_type="application/json" worked
                result_json = json.loads(response.text)
                # Validate expected keys
                if all(k in result_json for k in ["classification", "confidence", "reasoning"]):
                     return result_json
                else:
                    print(f"Error: LLM JSON response missing expected keys for {image_path}. Response: {result_json}")
                    return None
            except json.JSONDecodeError as e:
                print(f"Error: Invalid JSON response from LLM for {image_path}: {e}. Response text: {response.text}")
                # Attempt to clean common markdown issues if JSON parsing fails initially
                cleaned_text = response.text.strip().removeprefix("```json").removesuffix("```").strip()
                try:
                    result_json = json.loads(cleaned_text)
                    if all(k in result_json for k in ["classification", "confidence", "reasoning"]):
                        print("Successfully parsed JSON after cleaning markdown.")
                        return result_json
                    else:
                         print(f"Error: Cleaned LLM JSON response missing expected keys for {image_path}. Cleaned text: {cleaned_text}")
                         return None
                except json.JSONDecodeError:
                     print(f"Error: Still unable to parse JSON after cleaning for {image_path}.")
                     return None

            except Exception as e: # Catch other potential errors during processing
                 print(f"Error processing LLM response for {image_path}: {type(e).__name__} - {e}")
                 return None
        else:
            # Handle cases where the response might be empty but not necessarily an error object
            # Check for safety ratings or other indicators of blocked content
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                 print(f"Error: Prompt blocked for {image_path}. Reason: {response.prompt_feedback.block_reason}")
                 last_used[api_key] = datetime.now() # Mark key as used
                 return None
            else:
                print(f"Error: Empty or unexpected response from LLM for {image_path}. Response: {response}")
                return None


    except genai.types.generation_types.BlockedPromptException as bpe:
         print(f"Error: Prompt blocked for {image_path}. Reason: {bpe}")
         # Mark key as used even if blocked to potentially cycle
         last_used[api_key] = datetime.now()
         return None
    except Exception as e:
        # Catch potential API errors like quota exceeded, invalid API key etc.
        print(f"Error calling LLM for {image_path}: {type(e).__name__} - {e}")
        # If we hit a rate limit or other API error, mark key as recently used
        last_used[api_key] = datetime.now()
        return None

# --- Modified SAM Segmentation Function ---

def segment_image_with_sam(
    output_folder,
    roi_coordinates=None,
    zoom=18,
    batch=True,
    erosion_kernel=(3, 3),
    mask_multiplier=255,
    segment_image_output=None,
    perform_llm_classification=True # New parameter to control LLM classification
):
    """
    Segment a satellite image using SAM, save segments as PNGs, optionally classify
    them using an LLM, and add results to the GeoJSON. Ensures outputs use EPSG:4326.

    Parameters:
    - output_folder (str): Directory to save outputs (GeoTIFF, mask, and shapefile).
    - roi_coordinates (list, optional): ROI bounds as [min_lon, min_lat, max_lon, max_lat] in WGS84 (EPSG:4326).
    - zoom (int, optional): Tile zoom level. Default is 17.
    - batch (bool, optional): Enable batch segmentation for large images. Default is True.
    - erosion_kernel (tuple, optional): Kernel size for mask erosion. Default is (3, 3).
    - mask_multiplier (int, optional): Multiplier for mask values. Default is 255.
    - segment_image_output (str, optional): Directory to save individual segment PNG images.
                                            LLM classification requires this directory.
    - perform_llm_classification (bool, optional): Whether to classify each segment PNG using the LLM.
                                                  Requires `segment_image_output` to be set. Default is True.
    """

    # Validate parameters for LLM classification
    if perform_llm_classification and not segment_image_output:
        print("Warning: LLM classification requested but `segment_image_output` directory is not provided. Disabling LLM classification.")
        perform_llm_classification = False
    if perform_llm_classification and not llm_prompt:
         print("Warning: LLM classification requested but prompt file 'sam_prompt.txt' was not loaded. Disabling LLM classification.")
         perform_llm_classification = False
    if perform_llm_classification and not API_KEYS:
         print("Warning: LLM classification requested but no API keys are configured. Disabling LLM classification.")
         perform_llm_classification = False


    # Ensure output directories exist
    os.makedirs(output_folder, exist_ok=True)
    if segment_image_output:
        os.makedirs(segment_image_output, exist_ok=True)

    # Define output file paths
    image_file = os.path.join(output_folder, "satellite.tif")
    mask_file = os.path.join(output_folder, "segment_mask.tif") # Renamed for clarity
    geojson_file = os.path.join(output_folder, "segmented_parcels.geojson") # Renamed for clarity

    # Define target CRS
    target_crs = { "type": "name", "properties": { "name": "urn:ogc:def:crs:EPSG::3857" } } # WGS 84

    # Download map tiles and create a GeoTIFF
    # tms_to_geotiff assumes WGS84 bbox and typically outputs in the tile source's projection (often Web Mercator)
    # or attempts to match the bbox. We will ensure it's WGS84 afterwards if needed, but samgeo
    # usually handles this well when vectorizing based on the source image's CRS derived from the bbox.
    print(f"Downloading map tiles for ROI: {roi_coordinates} at zoom {zoom}...")
    tms_to_geotiff(output=image_file, bbox=roi_coordinates, zoom=zoom, source="Satellite", overwrite=True)
    print(f"Satellite image saved to {image_file}")

    # --- Verify and potentially enforce CRS for the downloaded image ---
    # Although samgeo usually handles this, let's add a check.
    # return None # Uncomment to make CRS error fatal

    # Initialize SAM and segment the image
    print("Initializing SAM model...")
    sam = SamGeo(
        model_type="vit_h", # Consider vit_l or vit_b for faster processing if acceptable
        # checkpoint="/path/to/your/sam_vit_h_4b8939.pth", # Optional: Specify local checkpoint if needed
        sam_kwargs=None,
    )
    print(f"Generating segmentation mask for {image_file}...")
    # sam.generate creates a mask based on the input image. It should preserve spatial reference.
    sam.generate(image_file, mask_file, batch=batch, foreground=True, erosion_kernel=erosion_kernel, mask_multiplier=mask_multiplier)
    print(f"Segmentation mask saved to {mask_file}")

    # Polygonize the raster data
    print(f"Converting mask {mask_file} to vector {geojson_file}...")
    # sam.tiff_to_vector should read the CRS from the mask_file (which comes from image_file)
    # and create a GeoJSON with that CRS.
    sam.tiff_to_vector(mask_file, geojson_file) # This saves the initial GeoJSON
    print(f"Initial vector data saved to {geojson_file}")

    # Read the generated GeoJSON and ensure CRS is correct
    try:
        geojson = gpd.read_file(geojson_file)
        print(f"Loaded {len(geojson)} segments from {geojson_file}")

        # --- Explicitly check and set CRS to EPSG:4326 ---

    except Exception as e:
        print(f"Error reading or setting CRS for GeoJSON file {geojson_file}: {e}")
        return None # Cannot proceed without the GeoJSON

    # --- Process and Classify Each Segment ---
    llm_classifications = []
    llm_confidences = []
    llm_reasonings = []

    if segment_image_output:
        print(f"Processing {len(geojson)} segments and saving PNGs to {segment_image_output}...")
        # Use tqdm for progress bar
        # Open the source image *once* outside the loop for efficiency
        try:
            with rasterio.open(image_file) as src:
                # Verify source CRS again just before cropping
              
                for idx, row in tqdm(geojson.iterrows(), total=len(geojson), desc="Processing Segments"):
                    segment_png_path = None
                    temp_tif = None # Define temp_tif here to ensure it's accessible in finally block
                    try:
                        # Ensure the geometry is valid
                        if not row['geometry'].is_valid:
                            print(f"Warning: Segment {idx} has invalid geometry. Attempting to buffer.")
                            # Attempt to fix with a zero buffer
                            geom = row['geometry'].buffer(0)
                            if not geom.is_valid or geom.is_empty:
                                 print(f"Error: Segment {idx} geometry could not be fixed. Skipping.")
                                 if perform_llm_classification:
                                     llm_classifications.append("Skipped - Invalid Geometry")
                                     llm_confidences.append(0)
                                     llm_reasonings.append("Original geometry was invalid and could not be fixed.")
                                 continue # Skip to next segment
                        else:
                            geom = row['geometry']

                        # Crop the raster using the segment's geometry
                        # Ensure geometry is mapped correctly if CRS mismatch occurred (though we try to enforce 4326)
                        out_image, out_transform = mask(src, [geom], crop=True, nodata=0, filled=True) # Ensure background is consistent

                        # Check if the cropped image is valid (non-zero dimensions)
                        if out_image.shape[1] == 0 or out_image.shape[2] == 0:
                            print(f"Warning: Segment {idx} resulted in an empty image after cropping (geometry might be too small or outside raster bounds). Skipping.")
                            if perform_llm_classification:
                                llm_classifications.append("Skipped - Empty Crop")
                                llm_confidences.append(0)
                                llm_reasonings.append("Segment geometry resulted in an empty image after mask/crop.")
                            continue # Skip to the next segment

                        out_meta = src.meta.copy()
                        out_meta.update({
                            "driver": "GTiff",
                            "height": out_image.shape[1],
                            "width": out_image.shape[2],
                            "transform": out_transform,
                            "nodata": 0, # Explicitly set nodata
                            "crs": src.crs # Preserve the CRS of the source image for the temp TIF
                        })

                        # Save the cropped raster as a temporary GeoTIFF
                        temp_tif = os.path.join(segment_image_output, f"temp_segment_{idx}.tif")
                        with rasterio.open(temp_tif, "w", **out_meta) as dest:
                            dest.write(out_image)

                        # Convert the GeoTIFF to PNG
                        with rasterio.open(temp_tif) as temp_src:
                            # Read only the first 3 bands (assuming RGB)
                            # Handle cases where source might have < 3 bands (e.g., grayscale)
                            num_bands = min(temp_src.count, 3)
                            segment_image_data = temp_src.read(list(range(1, num_bands + 1)))
                            if num_bands < 3:
                                # If less than 3 bands, duplicate the last band to make it RGB for PIL
                                print(f"Warning: Segment {idx} source has only {num_bands} bands. Creating RGB PNG.")
                                while segment_image_data.shape[0] < 3:
                                     segment_image_data = np.vstack([segment_image_data, segment_image_data[-1:, :, :]])


                        # Transpose to (height, width, channels) for PIL
                        segment_image_data = np.transpose(segment_image_data, (1, 2, 0))

                        # Ensure data is in uint8 format for PIL
                        if segment_image_data.dtype != np.uint8:
                             # Basic scaling if not uint8 (might need adjustment based on actual data range)
                             if segment_image_data.max() > 255:
                                 segment_image_data = (segment_image_data / segment_image_data.max() * 255).astype(np.uint8)
                             else:
                                 segment_image_data = segment_image_data.astype(np.uint8)


                        img = Image.fromarray(segment_image_data, 'RGB')
                        segment_png_path = os.path.join(segment_image_output, f"segment_{idx}.png")
                        img.save(segment_png_path)


                        # --- Perform LLM Classification ---
                        if perform_llm_classification and segment_png_path:
                            llm_result = call_llm(segment_png_path)
                            if llm_result:
                                llm_classifications.append(llm_result.get('classification', 'LLM Error - Key Missing'))
                                llm_confidences.append(llm_result.get('confidence', 0))
                                llm_reasonings.append(llm_result.get('reasoning', 'LLM Error - Key Missing'))
                            else:
                                llm_classifications.append("LLM Call Failed")
                                llm_confidences.append(0)
                                llm_reasonings.append("LLM API call or processing failed.")
                        elif perform_llm_classification: # Case where PNG path wasn't set (should not happen if logic is correct)
                             llm_classifications.append("Skipped - No PNG")
                             llm_confidences.append(0)
                             llm_reasonings.append("PNG file was not generated for this segment.")


                    except ValueError as ve:
                         # Catch specific rasterio mask errors like "Input shapes do not overlap raster."
                         print(f"Warning: Skipping segment {idx} due to geometry/mask error: {ve}")
                         if perform_llm_classification:
                             llm_classifications.append("Skipped - Geometry/Mask Error")
                             llm_confidences.append(0)
                             llm_reasonings.append(f"Rasterio mask error: {ve}")
                    except rasterio.errors.RasterioIOError as rio_err:
                         print(f"Warning: Skipping segment {idx} due to Rasterio IO error: {rio_err}")
                         if perform_llm_classification:
                             llm_classifications.append("Skipped - Rasterio Error")
                             llm_confidences.append(0)
                             llm_reasonings.append(f"Rasterio IO error: {rio_err}")
                    except Exception as e:
                        print(f"Error processing segment {idx}: {type(e).__name__} - {e}")
                        # Ensure lists stay synchronized even if an error occurs mid-segment processing
                        if perform_llm_classification and len(llm_classifications) < (idx + 1):
                             llm_classifications.append("Processing Error")
                             llm_confidences.append(0)
                             llm_reasonings.append(f"Error during segment processing: {e}")
                    finally:
                        # Remove the temporary GeoTIFF file if it exists
                        if temp_tif and os.path.exists(temp_tif):
                            try:
                                os.remove(temp_tif)
                            except OSError as ose:
                                print(f"Warning: Could not remove temporary file {temp_tif}: {ose}")

        except rasterio.RasterioIOError as e:
             print(f"FATAL ERROR: Could not open source image {image_file}. Cannot process segments. Error: {e}")
             return None # Cannot proceed if source image isn't readable
        except Exception as e:
             print(f"FATAL ERROR: An unexpected error occurred before segment processing loop: {e}")
             return None


        print("Finished processing segments.")

        # --- Add LLM results to GeoDataFrame ---
        if perform_llm_classification:
            print("Adding LLM classification results to GeoDataFrame...")
            # Ensure the lengths match before assigning columns
            if len(llm_classifications) == len(geojson):
                geojson['llm_class'] = llm_classifications
                geojson['llm_conf'] = llm_confidences
                geojson['llm_reason'] = llm_reasonings
            else:
                print(f"Error: Mismatch between number of segments ({len(geojson)}) and LLM results ({len(llm_classifications)}). LLM results not added.")
                # Optionally pad the lists if desired, but indicates an earlier error
                # pad_len = len(geojson) - len(llm_classifications)
                # if pad_len > 0:
                #     llm_classifications.extend(["Error - Mismatch"] * pad_len)
                #     llm_confidences.extend([0] * pad_len)
                #     llm_reasonings.extend(["Result list length mismatch"] * pad_len)
                #     geojson['llm_class'] = llm_classifications
                #     geojson['llm_conf'] = llm_confidences
                #     geojson['llm_reason'] = llm_reasonings
                #     print("Padded LLM results due to mismatch.")


    elif perform_llm_classification:
         print("LLM classification requested, but segment PNGs were not generated (`segment_image_output` not set).")


    # --- Save the final GeoJSON (with or without LLM results) ---
    # Ensure the CRS is correctly set before saving
    # if geojson.crs != target_crs:
    #     print(f"Re-confirming GeoDataFrame CRS is set to {target_crs} before final save.")
    #     geojson.crs = target_crs

    try:
        geojson.to_file(geojson_file, driver='GeoJSON')
        with open(geojson_file, 'r') as f:
            gj_content = json.load(f)
            gj_content['crs'] = { "type": "name", "properties": { "name": "urn:ogc:def:crs:EPSG::3857" } }
            with open(geojson_file, 'w') as f:
                json.dump(gj_content, f)
        print(f"Final GeoJSON (CRS: {geojson.crs}) saved to {geojson_file}")
    except Exception as e:
        print(f"Error saving final GeoJSON to {geojson_file}: {e}")


    print(f"\nScript finished. Main outputs are in {output_folder}.")
    if segment_image_output:
        print(f"Segment PNGs are in {segment_image_output}.")
    print(f"Final GeoJSON: {geojson_file}")
    if perform_llm_classification and 'llm_class' in geojson.columns:
        print("GeoJSON includes LLM classification columns.")
    elif perform_llm_classification:
        print("GeoJSON does *not* include LLM columns, likely due to processing errors or length mismatch.")


    return geojson

# --- Example Usage ---
if __name__ == "__main__":
    output_dir = "sam_output_4326" # Changed folder name slightly
    # Example ROI (replace with your actual coordinates) - WGS84 / EPSG:4326
    # Small area in Aathikkottai, Tamil Nadu
    roi_bounds = [78.0285045108849715, 8.6232691768964589, 78.0324210991931011, 8.6262826991796491] # [min_lon, min_lat, max_lon, max_lat]

    segment_image_output_dir = "sam_segment_images_4326" # Specify the output directory for segment images

    # Ensure the prompt file exists
    if not os.path.exists("sam_prompt.txt"):
         print("FATAL ERROR: sam_prompt.txt not found in the current directory. Please create it.")
         print("LLM classification will be disabled.")
         # Decide whether to exit or continue without LLM
         # exit() # Uncomment to stop execution if prompt is missing

    # Run the segmentation and classification
    # Set perform_llm_classification=True explicitly if you want it (it's the default now)
    # Set it to False if you only want segmentation and PNG generation
    final_geojson_data = segment_image_with_sam(
        output_dir,
        roi_bounds,
        zoom=18, # Increased zoom for potentially better detail
        segment_image_output=segment_image_output_dir,
        perform_llm_classification=True # Explicitly True
    )

    if final_geojson_data is not None:
        print("\n--- Sample of Final GeoDataFrame ---")
        print(final_geojson_data.head())
        print(f"GeoDataFrame CRS: {final_geojson_data.crs}")


        # Example: Accessing LLM results for the first few segments (if classification was done)
        if 'llm_class' in final_geojson_data.columns:
             print("\n--- LLM Classification Sample ---")
             print(final_geojson_data[['llm_class', 'llm_conf']].head())
        else:
             print("\nLLM classification columns not found in the final GeoDataFrame.")

    else:
        print("\nSegmentation process failed or returned None.")

    # --- Verification Step (Optional) ---
    # You can manually inspect the CRS of the output files:
    geojson_output_file = os.path.join(output_dir, "segmented_parcels.geojson")
    tif_output_file = os.path.join(output_dir, "satellite.tif")
    mask_output_file = os.path.join(output_dir, "segment_mask.tif")

    print("\n--- Verifying Output CRS ---")
    try:
        if os.path.exists(geojson_output_file):
            gdf_check = gpd.read_file(geojson_output_file)
            print(f"GeoJSON ({os.path.basename(geojson_output_file)}) CRS: {gdf_check.crs}")
            # Check the actual content for the CRS block
            with open(geojson_output_file, 'r') as f:
                gj_content = json.load(f)
                gj_content['crs'] = { "type": "name", "properties": { "name": "urn:ogc:def:crs:EPSG::3857" } }
                with open(geojson_output_file, 'w') as f:
                    json.dump(gj_content, f)
                if 'crs' in gj_content:
                    print(f"GeoJSON file 'crs' field: {gj_content['crs']}")
                else:
                    print("GeoJSON file does not contain a top-level 'crs' field (may rely on default WGS84).")

        if os.path.exists(tif_output_file):
             with rasterio.open(tif_output_file) as ds:
                 print(f"Satellite TIF ({os.path.basename(tif_output_file)}) CRS: {ds.crs}")
        if os.path.exists(mask_output_file):
             with rasterio.open(mask_output_file) as ds:
                 print(f"Mask TIF ({os.path.basename(mask_output_file)}) CRS: {ds.crs}")

    except Exception as e:
        print(f"Error during verification: {e}")

# --- END OF FILE new_sam.py ---