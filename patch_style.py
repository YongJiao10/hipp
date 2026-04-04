import json

label_list_file = "external/FASTANS/resources/PFM/priors/Kong2019/Kong2019_LabelList.txt"
style_json_file = "config/hipp_network_style.json"

with open(style_json_file, "r") as f:
    style = json.load(f)

# Collect existing names
existing_names = {v["name"]: k for k, v in style.items()}

# Get next id
next_id = max([int(k) for k in style.keys()]) + 1

with open(label_list_file, "r") as f:
    lines = f.readlines()

for i in range(0, len(lines), 2):
    name = lines[i].strip()
    rgba_line = lines[i+1].strip().split()
    if name not in existing_names:
        r, g, b, a = map(int, rgba_line[1:5])
        style[str(next_id)] = {
            "name": name,
            "rgba": [r, g, b, a]
        }
        print(f"Added {name} with id {next_id}")
        next_id += 1

with open(style_json_file, "w") as f:
    json.dump(style, f, indent=2)

