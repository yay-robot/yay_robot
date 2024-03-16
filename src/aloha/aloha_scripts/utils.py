import os, json

def get_auto_index(dataset_dir, dataset_name_prefix = '', data_suffix = 'hdf5'):
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for i in range(1000):
        if not os.path.isfile(os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i}.{data_suffix}')):
            return i
    raise Exception("Error getting auto index, or index is too big")

def get_task(task_name):
    try:
        with open("task_constants.json", "r", encoding='utf-8') as f:
            tasks = json.loads(f.read())
    except Exception as e:
        raise Exception("Error loading task_constants.")

    if task_name in tasks:
        return tasks[task_name]
    raise Exception("Task not specified in task constants file.")