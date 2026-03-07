---
date : '2025-11-28T21:06:20+05:45'
draft : True
title : 'Finetuning Large Language model for uncertainty-aware anwser generation.'
math : True
author : "Suyog Ghimire"
---
1. Introduction (Casual + Hook)

Start with a relatable scenario: maybe your chatbot confidently giving wrong answers.

Explain the problem simply: LLMs often hallucinate when unsure.

State what you did: “I fine-tuned a small LLM so that it can say ‘I don’t know’ when appropriate.”

2. Why It Matters (Educational)

Talk briefly about hallucinations in AI.

Explain the risks of overconfident models.

Mention real-world relevance (education tools, customer support bots, research assistants).

3. Approach (Technical but Accessible)

Data Preparation: How you created prompts where the model should respond “I don’t know.”

Fine-tuning: Keep it high-level, maybe mention tools/libraries (like Hugging Face, LoRA, etc.) without too much code.

Challenges: Overfitting, making it not too cautious, balancing “I don’t know” vs. helpful answers.

Use diagrams or simple tables to illustrate training examples.

4. Results / Observations

Share examples of before vs after.

Explain improvements and any surprising behavior.

Keep it light with screenshots or small snippets.

5. Lessons Learned (Casual + Educational)

Tips for small LLM fine-tuning.

Trade-offs: model size, inference speed, risk of over-saying “I don’t know.”

Broader implications: why handling uncertainty matters in AI.

6. Next Steps / Ideas for Readers

Encouraging readers to try it themselves.

Variations: confidence thresholds, reinforcement learning, multi-step reasoning.

Mention your idea for scaling or applying to other domains.

7. Conclusion

Wrap up casually, e.g., “Sometimes admitting you don’t know is the smartest answer. Turns out, the same is true for AI!”