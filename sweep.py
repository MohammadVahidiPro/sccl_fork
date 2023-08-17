metric = {
    'name': 'best/model/avg',
    'goal': 'maximize'
}

parameters_dict = {
    #'dataname':     {'values': ['searchsnippets', 'stackoverflow', 'biomedical']},
    #'batch_size':   {'values': [512, 400]},
    'eta':          {'values': [10,5]}
    }    

parameters_dict.update({
    'dataname':    {"value": "searchsnippets"},
    'bert':        {'value': 'minilm6'},                           #, 'distilbert']},
    'max_iter':    {'value':  1000},
    "batch_size":  {'value':  512},
    'log_mode':    {'value': 'offline'}
})

sweep_config = {
    'program': 'main.py',
    'method': 'grid',
    'metric': metric,
    'parameters': parameters_dict
}

import pprint, wandb

pprint.pprint(sweep_config)

sweep_id = wandb.sweep(sweep=sweep_config, project='sccl-2021')
print(f"sweep id: {sweep_id}")
print(f"sweep id: {sweep_id}")
print(f"sweep id: {sweep_id}")
print(f"sweep id: {sweep_id}")
print(f"sweep id: {sweep_id}")

wandb.agent(sweep_id=sweep_id)

print(f"sweep id: {sweep_id}")
print(f"sweep id: {sweep_id}")
print(f"sweep id: {sweep_id}")
print(f"sweep id: {sweep_id}")
print(f"sweep id: {sweep_id}")
