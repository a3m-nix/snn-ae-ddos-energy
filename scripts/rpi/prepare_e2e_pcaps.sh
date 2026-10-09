#!/bin/bash
set -e

cd /home/a3m-nix/snn-ae-final/pcap-e2e

for ds in edge_iiotset ciciot2023; do
    for k in normal attack; do
        editcap -r \
            "$ds/$k.pcap" \
            "$ds/${k}_e2e.pcap" \
            1500001-1700000
    done
done
