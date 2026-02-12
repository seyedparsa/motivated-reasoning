#!/bin/bash
squeue -u "${USER}" -o "%.18i %.20P %.35j %.2t %.10M"
