import pandas as pd
import ast

def write_counts_to_file(data_dict, filename, k=None):

    with open(filename, 'w') as f:
        for model_name, img_dict in data_dict.items():

            # Sort images of this model by prioritary (descending)
            sorted_items = sorted(
                img_dict.items(),
                key=lambda item: item[1].get('prioritary', 0),
                reverse=True
            )

            # Apply top-k
            if k is not None:
                sorted_items = sorted_items[:k]

            # Format top-k images for this model
            images_str = ", ".join(
                f"Name of the image: {img}: prioritary={counts.get('prioritary', 0)}, not-prioritary={counts.get('non_prioritary', 0)}"
                for img, counts in sorted_items
            )

            # Write one line per model
            line = f"{model_name}: {images_str}\n"
            f.write(line)

    print(f"Successfully wrote results for {len(data_dict)} models to {filename}")
    return filename

def generate_prediction_counts(df): 
    
    
    # Identify columns to process dynamically
    # We process everything that is NOT the 'gt' column and NOT the 'img' identifier
    target_cols = [col for col in df.columns if col not in ['gt', 'img']]
    
    master_results = {}
    
    print(f"Processing the following columns: {target_cols}")

    # 2. Iterate through each target column
    for col in target_cols:
        col_counts = {}
        
        for index, row in df.iterrows():
            img_name = row['img']
            cell_value = row[col]
            
            # 3. Parse string to list
            try:
                if isinstance(cell_value, str):
                    prediction_list = ast.literal_eval(cell_value)
                else:
                    # Handle cases where it might already be a list or NaN
                    prediction_list = cell_value if isinstance(cell_value, list) else []
            except (ValueError, SyntaxError):
                prediction_list = []

            # 4. Count objects
            p_count = 0
            np_count = 0
            
            for item in prediction_list:
                # Prediction format is (class_id, [box], score)
                # We take index 0 for the class_id
                try:
                    class_id = int(item[0])
                    
                    if class_id in PRIORITARY_CLASSES:
                        p_count += 1
                    else:
                        np_count += 1
                except (IndexError, ValueError):
                    continue
            
            # Store counts for this image
            col_counts[img_name] = {
                'prioritary': p_count, 
                'non_prioritary': np_count
            }
            
        # Add to master dictionary
        master_results[col] = col_counts
        
    return master_results