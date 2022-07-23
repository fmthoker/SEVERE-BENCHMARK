"""Reads action classes from a file and returns them as a list."""
from os.path import abspath, join

path = "data/ava/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt"
path = abspath(join(__file__, "..", "..", "..", path))


def read_label_map(label_map_path):
    """Reads a label map file and returns a dictionary mapping label names to
    
    Ref: https://stackoverflow.com/questions/55218726/how-to-open-pbtxt-file
    """

    item_id = None
    item_name = None
    items = {}
    
    with open(label_map_path, "r") as file:
        for line in file:
            line.replace(" ", "")
            if line == "item{":
                pass
            elif line == "}":
                pass
            elif " id:" in line:
                try:
                    item_id = int(line.split(":", 1)[1].strip())
                except:
                    import ipdb; ipdb.set_trace()
            elif " name:" in line:
                item_name = line.split(":", 1)[1].replace("'", "").strip()

            if item_id is not None and item_name is not None:
                items[item_name] = item_id
                item_id = None
                item_name = None

    return items


if __name__ == "__main__":
    label_map = read_label_map(path)
    labels = list(label_map.keys())
    
    with open(path.replace(".pbtxt", "_flat_list.txt"), "w") as file:
        for label in labels:
            file.write(f"{label}\n")
    import ipdb; ipdb.set_trace()
    print(label_map)
    