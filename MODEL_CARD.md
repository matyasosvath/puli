# Model Details

Hungarian Research Centre for Linguistics developed and released the Puli family of large language models (LLMs), a collection of pretrained and fine-tuned generative text models ranging in scale from ~400 million to ~7 billion parameters. Our fine-tuned LLMs, called Parancs PULI and PULI Llumix 32k Instruct, are optimized for dialogue use cases.

**Model Developers** Hungarian Research Centre for Linguistics

**Input** Models input text only.

**Output** Models generate text only.

**Model Architecture** PULI models are auto-regressive language models that uses transformer architecture. The tuned versions use supervised fine-tuning (SFT).


||Training Data|Params|Context Length|Words|
|---|---|---|---|---|
PULI 2 |*A collection of publicly and privately available data*<sup>1</sup>|354.6M|1024|32.2B


**Research Paper**. More information can be found in the following papers

1. Yang Zijian Győző, Dodé Réka, Ferenczi Gergő, and Váradi Tamás. “Jönnek a nagyok! BERT-Large, GPT-2 és GPT-3 nyelvmodellek magyar nyelvre.” Szeged, 2023.


## Intended Use

**Intended Use**. Puli models are intended for commercial and research use in Hungarian. Tuned models are intended for assistant-like chat, whereas pretrained models can be adapted for a variety of natural language generation tasks.

**Out-of-scope Uses**. Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in languages other than Hungarian. Use in any other way that is prohibited.**

## Ethical Considerations and Limitations

Puli models are a new technology that carries risks with use. Testing conducted to date has been in Hungarian, and has not covered, nor could it cover all scenarios. For these reasons, as with all LLMs, Puli models potential outputs cannot be predicted in advance, and the model may in some instances produce inaccurate, biased or other objectionable responses to user prompts. Therefore, before deploying any applications of Puli models, developers should perform safety testing and tuning tailored to their specific applications of the model.


## References

1. Yang Zijian Győző, Dodé Réka, Ferenczi Gergő, and Váradi Tamás. “Jönnek a nagyok! BERT-Large, GPT-2 és GPT-3 nyelvmodellek magyar nyelvre.” Szeged, 2023.
