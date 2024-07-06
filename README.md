# 🛢️ Database Gym 🏋🏻‍♂️
An end-to-end system for training RL agents to tune databases.

## Overview
The Database Gym (DBGym) is a research project from the CMU Database Group (CMU-DB) in the field of automated database tuning via reinforcement learning (RL). **Tuning** a database means selecting a configuration ([1] indexes, [2] system-wide knobs, and [3] per-query knobs) that optimizes the throughput or latency of a **workload** (a set of queries over a collection of data).

RL involves an **agent** performing **actions** against an **environment** which then gives **rewards** and **observations** back to the agent. When applying RL to database tuning, the actions are modifications to the database's configuration, the rewards are the throughput or latency (possibly estimated) of the workload under the new configuration. The observations may be things such as the current configuration or various execution statistics from running the workload.

In this workflow, a key challenge is in gathering the rewards and observations. Naively, this involves executing the full workload after every tuning action, but this introduces significant overhead. The DBGym project researches two directions to mitigate this overhead: [1] **approximating** the rewards and observations without executing the full workload and [2] extrapolating the information (rewards and observations) received about a single configuration to learn about **multiple configurations**. The first direction corresponds to the environment while the second direction corresponds to the agent, which is why DBGym is an **end-to-end** system.

## Architecture

## How to Run

## Cite