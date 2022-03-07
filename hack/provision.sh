#!/bin/bash
gcloud compute instances create instance-1 --project=stobias-dev --zone=us-central1-a --machine-type=n1-standard-16 --network-interface=network-tier=PREMIUM,subnet=default --maintenance-policy=TERMINATE --service-account=550789990239-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append --accelerator=count=1,type=nvidia-tesla-t4 --create-disk=auto-delete=yes,boot=yes,device-name=instance-1,image=projects/ml-images/global/images/c0-deeplearning-common-cu113-v20220227-debian-10,mode=rw,size=150,type=projects/stobias-dev/zones/us-central1-a/diskTypes/pd-balanced --no-shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring --reservation-affinity=any