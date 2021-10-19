import yaml
import os

scale = {
    "aloi": 10,
    "census": 1, # check
    "covtype": 1e-8,
    "glove": 10000,
    "lastfm": 100,
    "mnist": 100,
    "msd": 10,
    "shuttle": 100,
    "svhn": 10, #check
}

if not os.path.exists("definitions"):
    os.makedirs("definitions")

for ds in scale:
    for h in [0.01, 0.001, 0.0001, 0.00001]:
        h_str = f'{h:f}'.rstrip('0')
        with open(os.path.join("definitions", f'askit_{ds}_{h_str}.yaml'), "w") as f:
            id_tol = [scale[ds] * h, scale[ds] * h / 10]
            oversampling = 2
            max_points = [512, 2048]
            k = [100]
            id_rank = [512]
            skel_targets = [2]
            num_uniform_required = [0]
            min_skel_level = [2, 10, 20]


            d = {
                "askit" : {
                    "constructor": "Askit",
                    "wrapper": "askit",
                    "docker": "deann-experiments-askit",
                    "separate-queries": True,
                    "query": [[k, id_tol, max_points, [oversampling], id_rank, skel_targets, num_uniform_required, min_skel_level]]
                }
            }

            yaml.dump(d, f)
