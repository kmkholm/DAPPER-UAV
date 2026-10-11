# DAPPER-UAV

Code and data for the paper **"DAPPER: Differential-Attention, Path-Personalised Federated Inference of Drone Flight Phase from Encrypted Cellular Command-and-Control Traffic"**.

The repository contains everything needed to regenerate the dataset in GNS3, to train and evaluate DAPPER and the federated baselines, and to rebuild every table and figure of the paper from the experiment logs.

## Contents

| Folder | Content |
|---|---|
| `emulation/` | GNS3 testbed: Docker image (ArduPilot 4.7.1 SITL, WireGuard, IPsec, tcpdump), node roles (`node.py`), MAVLink forwarder and 5 Hz ground truth (`uplink.py`), mission supervisor (`gcs.py`), cross-traffic (`bg.py`), project builder (`topology.py`), data fetcher, screenshot scripts |
| `code/` | Packet processing (`process.py`, `packets.py`), windows and splits (`data.py`), model (`model.py`), federated training (`train.py`), experiment grid (`run_queue.py`), stream evaluation, traffic-shaping defences (`defenses.py`), table generation (`build_tables.py`) |
| `figures/` | Scripts for all figures |
| `data/proc/` | Processed dataset: one `<site>_<sysid>.npz` per drone (100 ms bin features, per-bin flight-phase labels, mission index) and one `<site>_<sysid>_pk.npz` per drone with every captured outer packet (timestamp, IP length, direction) |
| `results/runs/` | Result files (JSON) of all 128 runs reported in the paper |

## Dataset

24 drones (12 ArduCopter multirotors over WireGuard, 12 ArduPlane VTOL aircraft over IPsec ESP) on eight emulated cell sites (5G, LTE, congested LTE, LEO and GEO satellite backhaul), 22.7 h of flight, 3.2 M encrypted packets. Labels come from the drones' own state logs and are independent of the traffic. Only packet headers of the encrypted tunnels were captured; no payload is stored.

Phase codes: multirotor `0 ground, 1 ascent, 2 hover, 3 cruise, 4 orbit, 5 tele-operation, 6 descent`; VTOL `0 ground, 1 VTOL climb, 2 VTOL descent, 3 cruise, 4 loiter, 5 tele-operation`; `-1` = no ground truth.

## Public over-the-air captures

`code/public_pinch.py` converts the public PiNcH dataset (Sciancalepore et al., Computer Networks 2020; download `data.zip` from https://cri-lab.net/pinch into `data/public_pinch/release`) into the same per-drone files, and `results/public/` holds the runs reported in the paper:

```bash
python code/public_pinch.py data/public_pinch/release data/public_pinch/proc
python code/train.py --net P --method local --task fine --seed 1 --rounds 20 --steps 50 --proc data/public_pinch/proc --out results/public
python code/train.py --net Q --method local --task fine --seed 1 --rounds 20 --steps 50 --T 64 --proc data/public_pinch/proc --out results/public
```

## Reproducing the results

```bash
pip install -r requirements.txt
cd code
python train.py --net A --method dapper --task op3 --seed 1 --rounds 20 --steps 50   # one run
python run_queue.py                                                                  # full grid (128 runs)
python eval_stream.py --net A --task op3 --json ../results/runs/stream_At.json
python build_tables.py                                                               # tables and numbers of the paper
cd ../figures && python results_figs.py && python repr_figs.py xai
```

Tasks: `op3` operational state (ground / autonomous flight / tele-operation), `op4` operational phases, `fine` kinematic phases. Methods: `local, central, fedavg, fedprox, fedbn, dapper` and the ablations `dapper_softmax, dapper_nogate, dapper_cal, dapper_prompt, dapper_nost, dapper_nobins, dapper_pk`. Defences: `--defense pad256 | pad512 | jitter200 | const`.

## Regenerating the dataset

Requires GNS3 2.2 with the GNS3 VM (Docker).

```bash
cd emulation
python build_image.py          # builds dapper-node:latest inside the GNS3 VM
python topology.py             # creates and starts the 44-node project
python fetch_data.py           # downloads captures and ground truth
cd ../code && python process.py ../data/raw ../data/proc && python packets.py ../data/raw ../data/proc
```

The GNS3 server password is read from the local `gns3_server.ini`; the VM address and login can be set with `GNS3VM_IP`, `GNS3VM_USER`, `GNS3VM_PW`.

## License

MIT. Author: Dr. Mohammed Tawfik (kmkhol01@gmail.com, ORCID 0009-0004-7622-5991).
