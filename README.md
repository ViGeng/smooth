# SMOOTH: Scalable Multitask Offloading with Backbone Sharing

This is the official implementation of the paper "SMOOTH: Scalable Multitask Offloading with Backbone Sharing" accepted by IFIP Networking 2026.

Repository structure:
- `heads-training`: Code for training the task-specific heads of the model.
- `offload`: Code for offloading system running on our testbed.
  - `server`: server-side code for handling offloaded tasks, including task scheduling and execution.
  - `client`: client-side code for offloading tasks to the server, including task preparation and communication with the server.
  - `utils`: Utility functions or shared code used by both server and client.
- `scripts`: Scripts for automating tasks such as data preprocessing, model training, and evaluation.
- `data`: Directory for storing raw data or processed data.

## Status

It takes a while to clean up the code and prepare the documentation for a project lasting several years, so we are releasing the code in stages. Below is the current status of each module:

| Module | Status |
| --- | --- |
| Heads Training | Released✅ |
| Raw/Processed Data | Released✅ |
| Offloading System | coming soon⏳ This is planned to be released together with our new paper based on this offloading system. Hopefully by June! |

## Citation

If you find our work useful in your research, please consider citing our paper:

TBD