# Experiments with Transformers

- Investigate Scaling laws (Chinchilla, Kaplan laws,...)
- Implement different MLA, GQA, MQA, SWA, etc.
- Implement KV caching
- Understand learning rate scheduling
- Implement MoE
- Investigate the effects of sequence packing in pretrainig
    - Currently I only implement truncation and padding - that is inefficient
- Experiment with layer norm position - what's its effect?

# Training infra

- 2x A100 80GB available in k8s cluster
- Run training script in a kubernetes job with custom container image and push artifacts to s3 container
