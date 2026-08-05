#!/bin/sh
# Minimal Lambda Cloud API helper. Requires ~/.lambda-api-key (chmod 600).
set -e
KEY=$(cat "$HOME/.lambda-api-key")
API=https://cloud.lambdalabs.com/api/v1
case "$1" in
  types)     curl -su "$KEY:" "$API/instance-types" | python3 -m json.tool ;;
  ls)        curl -su "$KEY:" "$API/instances" | python3 -m json.tool ;;
  keys)      curl -su "$KEY:" "$API/ssh-keys" | python3 -m json.tool ;;
  launch)    [ -n "$3" ] || { echo "usage: $0 launch REGION SSHKEY" >&2; exit 1; }
             curl -su "$KEY:" -X POST -H 'Content-Type: application/json' \
               -d "{\"region_name\":\"$2\",\"instance_type_name\":\"gpu_1x_a100\",\"ssh_key_names\":[\"$3\"]}" \
               "$API/instance-operations/launch" | python3 -m json.tool ;;
  terminate) [ -n "$2" ] || { echo "usage: $0 terminate INSTANCE_ID" >&2; exit 1; }
             curl -su "$KEY:" -X POST -H 'Content-Type: application/json' \
               -d "{\"instance_ids\":[\"$2\"]}" \
               "$API/instance-operations/terminate" | python3 -m json.tool ;;
  *) echo "usage: $0 types|ls|keys|launch REGION SSHKEY|terminate ID" >&2; exit 1 ;;
esac
