from util import general_utils

def get_data_files(data_dirs, path_to_data, temporalROI="temporalROI", max_video_length=None):
    data_dirs = [f"{path_to_data}/{data_dir}" for data_dir in data_dirs]
    
    data = dict()
    
    for data_dir in data_dirs:
        line = general_utils.read_lines(f"{data_dir}/{temporalROI}.txt")[0]
        parts = line.split(" ")
        start_frame = int(parts[0])
        end_frame = int(parts[1])
        
        if max_video_length is not None:
            end_frame = min(start_frame + max_video_length - 1, end_frame)
        
        if data_dir not in data:
            data[data_dir] = dict()            
        
        for i in range(start_frame, end_frame + 1):
            if "inputs" not in data[data_dir]:
                data[data_dir]["inputs"] = list()
                data[data_dir]["groundtruths"] = list()
                
            data[data_dir]["inputs"].append(f"{data_dir}/input/in{'{:06d}'.format(i)}.jpg")
            data[data_dir]["groundtruths"].append(f"{data_dir}/groundtruth/gt{'{:06d}'.format(i)}.png")
            
    return data
        
        