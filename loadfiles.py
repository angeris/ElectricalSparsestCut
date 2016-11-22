from collections import defaultdict
import os

def load_dict_network(dataset_dir='./dataset'):
    """Returns a dictionary of connections
    """
    all_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.egonet')]
    all_connections = defaultdict(set)
    for curr_f in all_files:
        f = open(curr_f)
        for line in f:
            f_split = line.strip().split(':')
            curr_key = int(f_split[0])
            all_friends = f_split[1].strip().split()
            for friend in all_friends:
                friend = int(friend)
                all_connections[curr_key].add(friend)
    return all_connections


print load_dict_network()
