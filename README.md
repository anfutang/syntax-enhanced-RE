# syntax-enhanced-RE
codebase for syntax-enhanced finetuning on relation extration tasks.
By adding "syntactic" loss terms to the original relation classification loss and fine-tune the new multi-task neural network, we suppose to build a *syntax-driven* neural network in which syntactic information is added in an implicit way without modifying pre-trained encoder architecture or introducing extra layers.

Advantage of our network are:
- no extra layer (fine-tuning purely from pre-trained checkpoints)
- no modification on encoder network architecture

