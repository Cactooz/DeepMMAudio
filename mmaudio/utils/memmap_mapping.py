import csv

def map_index(source_path, target_path, output_path):
    id_index = {}
    
    try:
        with open(target_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            _ = next(reader, None)
            
            for index, row in enumerate(reader):
                if row:
                    row_id = row[0]
                    id_index[row_id] = index
                    
        print(f"Found {len(id_index)} IDs in target file")
        
        mapped = 0
        missing = 0
        
        with open(source_path, 'r', newline='', encoding='utf-8') as f_in, \
             open(output_path, 'w', newline='', encoding='utf-8') as f_out:
            
            reader = csv.reader(f_in, delimiter='\t')
            writer = csv.writer(f_out, delimiter='\t')
            _ = next(reader, None)
            
            for _, row in enumerate(reader):
                if row:
                    source_id = row[0]
                    
                    if source_id in id_index:
                        target_index = id_index[source_id]
                        writer.writerow([target_index])
                        mapped += 1
                    else:
                        print(f"Warning: '{source_id}' not found in target file")
                        writer.writerow([""]) 
                        missing += 1

        print(f"Output saved to: {output_path}")
        print(f"Mapped IDs: {mapped}")
        if missing > 0:
            print(f"Missing IDs: {missing}")

    except Exception as e:
        print(f"An error occurred when processing the files: {e}")

if __name__ == "__main__":
    source_file = 'data_v1-16-memmap_vgg-train-depth.tsv'
    target_file = 'data_v1-16-memmap_vgg-train.tsv'
    output_file = 'memmap-mapping-train.tsv'
    
    map_index(source_file, target_file, output_file)
