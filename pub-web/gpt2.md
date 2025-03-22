# A Vector-Level View of GPT-2

Computational infographic of vector-wise inference in a decoder-only transformer

With annotations based on Anthropic’s [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/)  
and Neel Nanda’s [Comprehensive Mechanistic Interpretability Explainer & Glossary](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J)

Jeremy Dolan, December 2024, v1.0

This diagram situates some of the core insights from the mechanistic interpretability research program in a visual walkthrough of a transformer model.

# Introduction

This graphic situates some of the core insights from the mechanistic interpretability research program in a visual walkthrough of a transformer model. The goal is to blend...

1. A vector-level computational graph

Neural networks are typically implemented as a series of *tensor operations* because modern tooling is highly optimized for the parallelized matrix multiplications that tensors facilitate. But the most computationally efficient way to *code* a neural network isn’t necessarily the best way to understand how it works. Vectors (aka embeddings) are the fundamental information-bearing units in transformers, and are—with few exceptions—operated on completely independently. Discussions framed in terms of \[batch × head × position × d\_head\] tensors, where thousands of high-dimensional vectors are packed together, can lose focus on how information actually flows through the model.

Implementations sometimes even permute the computational structure of the architecture for efficiency. For example, [the original transformers paper](https://arxiv.org/abs/1706.03762) describes multi-headed attention as involving a concatenation of the result vectors from each head, which is then projected back to the residual stream. Implementations and discussion since has largely adhered to this structuring. But concatenation is an unprincipled operation that obscures the natural way information flows through attention heads: result vectors are independently meaningful, and they can be directly and independently projected back to the residual stream without any concatenation operation.

	with,

2. A mechanistic interpretability infographic

Existing work (such as Anthropic’s excellent [Transformer Circuits](https://transformer-circuits.pub/) thread) is weighty, and our understanding is rapidly evolving. A good primer might help people bootstrap into this important research program.

The intended audience already has a rough understanding of the transformer architecture. If a refresher is needed, I recommend Jay Alamar’s [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/). Note that my diagram depicts a decoder-only model (GPT-2 124M, a common reference model for interpretability work) rather than the original encoder-decoder architecture depicted in Alamar’s piece.

Created for BlueDot Impact’s AI Safety Fundamentals’ [AI Alignment Course](https://aisafetyfundamentals.com/alignment/).

# Infographic

## Tokenization, embedding, and the residual stream

### Tokenizer

Decomposes an input string into substring tokens based on a predefined dictionary, and outputs the index values for those tokens. The tokenizer is an input pre-processor rather than part of the transformer architecture itself.

GPT-2 uses a [BPE tokenizer](https://jeremydolan.net/transformer-view/assets/BPE_tokenization/) with tokens for all 256 byte values, 50,000 [merges](https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe), and a special \<|endoftext|\> token giving a total vocabulary \\(n\_{vocab}\\) of 50,257. (Methods that display tokens [replace](https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/encoder.py#L25) control and whitespace characters with visible alternatives; thus the space, U+0020, in our example input is shown as “Ġ”, U+0120.)

### Embedding weight matrix

\\(n\_{vocab} \\times d\_{model}\\) \= 50,257 \\(\\times\\) 768

A learned lookup table mapping token IDs to token embeddings. Embeddings are high-dimensional dense vectors which the model learns for each token ID during training. The embeddings capture some generalized (roughly: context independent) meaning for each token. How is this meaning represented in the embedding? A central tenet of mechanistic interpretability is that [meaningful features are represented by directions in the embedding space](https://distill.pub/2020/circuits/zoom-in/#claim-1). However, because the embedding space has no privileged basis, individual components of the learned embeddings are not independently meaningful.

For GPT-2 124M, the embedding space is 768-dimensional. A 768-component vector for each token in the model’s 50,257 token vocabulary means that \\(W\_E\\) comprises 38,597,376 parameters—31% of the model’s total\! (This ratio [decreases significantly](https://jeremydolan.net/transformer-view/assets/parameter-allocation/) in larger models.)

### Positional embedding weight matrix

\\(n\_{context} \\times d\_{model}\\) \= 1,024 \\(\\times\\) 768

A learned lookup table mapping sequence positions to positional encodings. A challenge for the transformer architecture is that it is “order invariant”: although the token embeddings are typically stored in order in a tensor, this ordering is not exposed to any of the model’s computation. But for language, order matters.

There are many approaches to adding position information (absolute or relative) to the token embeddings (or to the attention heads directly). GPT-2 simply parameterizes it, allowing the model to figure out for itself how to represent sequence order. Each row \\(i\\) of \\(W\_{P}\\) contains a learned “time signal” that is added to token embedding \\(i\\), thereby marking its position. This means that the size of this matrix must be defined at training time, effectively constraining the maximum sequence length (“context length”) that the model can process (1,024 for GPT-2).

### Input embeddings

\\(n\_{tokens} \\times d\_{model}\\) \= 3 \\(\\times\\) 768

\\(d\_{model}\\)-dimensional vectors, each the sum of a token embedding (encoding context-independent semantic meaning) and a positional embedding (encoding sequence position). The input embeddings are the model’s initial representation of the “meaning” of each token, and they will be contextualized, refined, and otherwise modified by each sublayer as they proceed through the residual stream.

Scale note: only \\(\\frac{1}{6}th\\) (128 boxes) of each vector’s 768 components are shown. Cf. the embedding size of foundation models in 2024, typically about \\(2^{14}\\) \= 16,384.

### Residual stream

\\(n\_{tokens} \\times d\_{model}\\) \= 3 \\(\\times\\) 768

Carries the embeddings to each sublayer of the model, with each sublayer applying an “edit.” The residual stream is the central [“communication channel”](https://transformer-circuits.pub/2021/framework/#residual-comms) in a transformer, providing a \\(d\_{model}\\)-dimensional latent space for each token. The residual stream does not process any information itself, but instead serves as the conduit through which all sublayers communicate.

Each attention and MLP sublayer “reads” in the residual stream, performs computations, and then “writes” a result back using element-wise addition. The residual stream is only modified by these addition operations; it is otherwise a series of end-to-end skip connections from the embedding to the unembed.

This linear, additive structure has several consequences:

1. Gradient Flow: During backpropagation, addition preserves gradient magnitudes. This allows the full gradient to reach every sublayer without diminishing, preventing “vanishing gradients” and making it feasible to train very deep models.  
2. Information Persistence: During the forward pass, the residual connections prevent over-degradation of early information. Thus, input embeddings and early layer outputs generally remain accessible to later layers unless [actively deleted](https://transformer-circuits.pub/2021/framework/#d-footnote-7-listing) by the model.

Persistence is further enhanced by a model learning to communicate through specific subspaces within the residual stream: a single layer [“can send different information to different layers by storing it in different subspaces.”](https://transformer-circuits.pub/2021/framework/#subspaces-and-residual-stream-bandwidth) This allows the model to optimize information routing and implement complex computational circuits. For example, some computation might go from an attention head in layer 2, through an MLP in layer 5, to an attention head in layer 8, and so on.

Accordingly, we can conceptualize transformers as implementing computational functions which correspond to particular paths through the model. Each path between communicating components (embedding, attention heads, MLP layers, and unembedding) is free to use its own encoding, and all of these encoded messages between components are superimposed in the residual stream. This makes the residual stream itself not a particularly tractable target for interpretability work: [“Rather than analyze the residual stream vectors, it can be helpful to decompose the residual stream into all these different communication channels, corresponding to paths through the model.](https://transformer-circuits.pub/2021/framework/#conceptual-take-aways)”

Further interpretability considerations:

* The residual stream is “the only way that information can move between layers, and so [the model needs to find a way to store all relevant information in there](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=DHp9vZ0h9lA9OCrzG2Y3rrzH).” The embedding space is a bottleneck, and high demand likely leads to significant [superposition](https://transformer-circuits.pub/2021/framework/#subspaces-and-residual-stream-bandwidth).  
* Although the residual stream was [predicted to have no privileged basis](https://transformer-circuits.pub/2021/framework/#def-privileged-basis), in practice some basis alignment has been observed. Recent work suggests [per-dimension normalizers in the Adam optimizer](https://transformer-circuits.pub/2023/privileged-basis/) are the most likely cause.

### Layer Normalization

Norms each residual stream vector (independently) to have mean 0 and variance 1, then scales and shifts each component \\(i\\) of the normalized vector by a learned scaling factor (\\(\\gamma\_{i}\\)) and shifting factor (\\(\\beta\_{i}\\)). Each LayerNorm \\(n\\) in the model learns and applies scaling and shifting vectors \\(\\gamma^n\\) and \\(\\beta^n\\), both of size \\(d\_{model}\\) \= 768\.

Purpose: The initial normalization promotes consistent distributions of input to a layer by reducing “covariate shift” [(Ba et al., 2016\)](https://arxiv.org/abs/1607.06450), thereby stabilizing gradients and speeding up training. The parameterized scale and shift then return some flexibility to the model, allowing it to optimize the transformation for each layer.

Implications: In GPT-2, LayerNorm mediates every “read” of the residual stream (at the start of each attention or MLP sublayer, and before Unembed). Because LayerNorm is a non-linear transformation, it complicates interpretability, which [prominent researchers](https://youtu.be/ll0oduwDEwI?t=3171) have [expressed frustration about](https://youtu.be/bOYE6E8JrtU?t=3525).

In principle, the non-linearity in LayerNorm can perform useful computation. In practice, however, it is an open question whether LayerNorm performs any “meaningful” computation within transformers, or if it is merely an artifact of the training process. Researchers have trained more interpretable models without LayerNorm, and [a recent preprint](https://www.arxiv.org/abs/2409.13710) found that GPT-2’s LayerNorms can be removed with only minor performance degradation after fine-tuning, suggesting it does not play a crucial role at inference.

[RMSNorm](https://arxiv.org/abs/1910.07467) has largely replaced LayerNorm in more recent models.

## Attention Block

### Attention head

Each attention head creates three projections of the residual stream which are used to select information to move between positions. Attention heads are the only place in the entire architecture where information can move between positions. The \\(n\_{head}\\) attention heads (12 in GPT-2) operate [independently and in parallel](https://transformer-circuits.pub/2021/framework/index.html#architecture-attn-independent): for each position, a QK circuit determines which positions to read from, and an OV circuit determines which information is copied.

### Query, Key, Value matrices

\\(d\_{model} \\times d\_{head}\\) \= 768 \\(\\times\\) 64 (each)

Each attention head \\(h\\) in each attention sublayer \\(l\\) has its own Query, Key, and Value weight matrices \\(W^{l,h}\_x\\) and bias vectors \\(b^{l,h}\_x\\). (In practice these are grouped in per-sublayer matrices for efficiency.) Each position in the residual stream is independently transformed by each of these matrices to form Query, Key, and Value projections. In effect, these learned matrices define [small, \\(d\_{head}\\)-dimensional subspaces](https://transformer-circuits.pub/2021/framework/index.html#subspaces-and-residual-stream-bandwidth) that will be read in from each position, for that head.

### Query, Key, and Value projections

\\(n\_{tokens} \\times d\_{head}\\) \= 3 \\(\\times\\) 64 (each)

### Causal attention

Attention in a transformer decoder is *causal*: each position can only attend to itself and *prior* positions in the sequence. Thus, information can only move forward. This greatly improves training efficiency. It is typically implemented by computing the dot-product of *all* Query/Key pairs and then masking the forward-looking half, but it is displayed here conceptually, with Querys interacting only with Keys at their own and prior positions. *This dot product, and the weighted sum below, are the only places in the model where positions interact.*

### Attention pattern

A triangular matrix produced by causal attention, used to take a weighted sum of the value vectors from each position.

### Result vectors

\\(n\_{tokens} \\times d\_{head}\\) \= 3 \\(\\times\\) 64

Because the ‘weights’ from softmax are a probability distribution, this is effectively a weighted average of Value projections.

### Output vectors

\\(n\_{tokens} \\times n\_{head} \\times d\_{model}\\) \= 3 \\(\\times\\) 12 \\(\\times\\) 768

Each attention head produces an output vector for each position, and these are added back to their respective position in the residual stream.

### Output weight matrix

\\(d\_{head} \\times d\_{model}\\) \= 64 \\(\\times\\) 768

Projects the result vectors (themselves weighted sums of the value projections) for each position back into the embedding space.

\\(W\_O\\) is depicted here as a per-attention-head transformation yielding per-attention-head output vectors. However, the projection back to embedding space is typically implemented as a per-layer transformation for computational efficiency: the weighted sums for each position from each head are stacked (concatenated) to make a Frankensteinian \\(d\_{model}\\)-dimensional vector, which is transformed all at once with \\(W^1\_O\\). Here, we instead take \\(d\_{head}\\)-sized slices of \\(W\_O^1\\) for each head. This is [mathematically equivalent](https://transformer-circuits.pub/2021/framework/index.html#architecture-attn-independent) and provides a per-head output which can be a locus of interpretation.

### An attention sublayer can be conceptualized as [two largely independent operations](https://transformer-circuits.pub/2021/framework/index.html#splitting-attention-head-terms-into-circuits)…

#### QK circuit

The QK circuit determines *where* information will be copied from, for each position. In GPT-2’s decoder-only architecture, attention is “causal”: information can only flow forward, or put another way, each position can only attend to itself and prior positions. For each input vector, the QK circuit calculates an attention score for positions in the sequence up to and including the input vector itself.

Attention scores are usually conceptualized as the alignment (a scaled dot product) of a destination’s “query” projection (\\(W\_Q\\mathbf{d}\\)) and a source’s “key” projection (\\(W\_K\\mathbf{s}\\)). But in an attention head \\(W\_Q\\) and \\(W\_K\\) always function together, so what determines the attention scores is the combined matrix \\(W\_Q^T W\_K\\). If you double \\(W\_Q\\) and half \\(W\_K\\), or rotate \\(W\_Q\\) one way and \\(W\_K\\) inversely, the attention scores do not change. Thus, the query and key projections are intermediary constructs of the attention calculation and unlikely to be independently meaningful.

Note: in later layers, the QK circuit is not comparing the original embeddings, but rich representations that have accumulated information from many other positions. This allows for complex attention patterns to emerge, such as attending to vectors based on abstract semantic or syntactic relationships.

#### OV circuit

The OV circuit determines *what* information will be copied from each position when it is read (\\(W\_V)\\) in an attention head, and how to write that information back to the embedding space (\\(W\_O\\)). Similar to the QK circuit, \\(W\_O\\) and \\(W\_V\\) always function together: each residual stream vector is transformed by the combined \\(d\_{model} \\times d\_{model}\\) matrix \\(W\_O W\_V\\). The result is copied to all subsequent positions and scaled by the corresponding value in the attention pattern. As before, the value projections are unlikely to be independently meaningful. The locus of interpretability should be the OV circuit as a whole, which takes in \\(n\_{tokens}\\) vectors from the residual stream and outputs \\(n\_{tokens}\\) new vectors to add to each position.

Note: attention is usually distributed over many positions, so each position will generally read in information from many others. After several layers, residual stream vectors will have collected a lot of information from other positions. This means that in later layers, if one position appears to be “reading” from another, the information being transferred may not be related to that positions original embedding.

## MLP Block

### MLP sublayer

Multi-layer perceptrons (alternatively, ‘feed-forward networks’) perform computation on each embedding in the residual stream. While attention sublayers move information between positions, MLP sublayers process the information that has been accumulated at each position. In an MLP each embedding is transformed *independently* (no information moves between positions) and by the same set of weights. Each embedding is normed before undergoing an expansion in dimension, a non-linear activation, and a contraction to the original space. The result is added back to the corresponding embedding in the residual stream. This sequence of linear map, non-linearity, linear map, is extremely flexible and can (with sufficient width) [approximate any function](https://link.springer.com/article/10.1007/BF02551274).

The MLPs account for roughly [two-thirds](https://docs.google.com/spreadsheets/d/1Vm6F41W30fOT_XtpRkUzStiVAPcHL-QxMFoahZAuAuQ/edit?usp=sharing) of the trainable parameters in a typical transformer, and the structure of the computation here is quite simple. But interpreting what these computations are doing has been a challenge. MLPs have been implicated in: [storage and retrieval of unsystematized facts](https://www.alignmentforum.org/posts/iGuwZTHWb6DFY3sKB/fact-finding-attempting-to-reverse-engineer-factual-recall) (“The capital of France is…”), non-linear feature recombination (boolean AND or XOR), abstractions and hierarchical features (France → nation state), memory management (amplifying or attenuating existing representations), and more.

### Up projection weight matrix

\\(d\_{model} \\times d\_{mlp}\\) \= 768 \\(\\times\\) 3,072

Maps the vectors from the residual stream to a higher dimensional latent space. The increased dimensionality allows the model to represent more interactions. The [original transformers paper](https://arxiv.org/abs/1706.03762) used \\(d\_{mlp} \= 4 \\times d\_{model}\\) and GPT-2 (and most other models) have copied this design choice.

A simplified way of conceptualizing this projection is to think of each of the 3,072 *columns* of this matrix as a probe for some feature or combination of features in the input vector. Each column (*i.e.*, a feature being probed) is dotted with the input vector, yielding a similarity score which tells us the degree to which that vector exhibits that feature. This score will then be used as a scaling factor during down-projection. (This simplification ignores the phenomenon of [superposition](https://transformer-circuits.pub/2022/toy_model/) which allows many more features than dimensions to be represented but greatly complicates interpretability.)

### Gaussian Error Linear Unit

The activation function in MLP sublayers is the central computational non-linearity of a transformer, allowing the model to represent non-linear relationships and approximate arbitrary functions. (LayerNorm and softmax are nonlinear, but they function as normalizing steps.)

GPT-2 uses [GELU](https://arxiv.org/abs/1606.08415), a smooth version of ReLU which is arguably more principled and demonstrably better performing. (More precisely, GPT-2 [uses](https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/model.py#L25) a tanh approximation of GELU.) GELU’s gating confers a [privileged basis](https://transformer-circuits.pub/2021/framework/index.html#def-privileged-basis) to the activation space, making the activations a fruitful target for interpretability work.

### Down projection weight matrix

\\(d\_{mlp} \\times d\_{model}\\) \= 3,072 \\(\\times\\) 768

Maps the activations in the MLP’s internal representation space back down to the model’s embedding space, so that the output vectors can be added back to the residual stream.

To continue our simplified conception, we can think of each of the 3,072 *rows* of this matrix as directions (*i.e.*, a feature or combination of features) to potentially be copied to the residual stream. Each row (potential output) is scaled by the corresponding activation from the up-projection’s “probing,” and then written back to the residual stream for this position. So, roughly, the up-projection asks the vector whether it has some feature (*e.g.*, representing the country of France), and the down-projection tells the model what to add to the representation if so (*e.g.*, “capital=Paris”, “currency=Euro”, “population=68 million”, “official language=French”)

## Back to the residual stream

**Layers 2–12 are elided here.** They are identical in structure to layer 1 (an attention sublayer followed by an MLP sublayer) but have their own learned weights.

During inference, **we only need to unembed and decode the final position**, because we only care about the final position’s next-token prediction, and that prediction no longer depends on any further interactions with earlier positions.

### Unembedding

The final residual stream vector is transformed by the unembedding weights, producing a \\(n\_{vocab}\\)-length list of similarity scores (output logits), one for each token in the model’s vocabulary.

The unembedding operation is a linear transformation, but it is best understood geometrically: effectively, each output logit is computed as the dot product between the final residual stream vector and the learned embedding for each token in the vocabulary. Thus the logits represent how ‘similar’ the final residual stream vector is to each token’s representation in the embedding space.

### Transposed embedding weight matrix

\\(d\_{model} \\times n\_{vocab}\\) \= 768 \\(\\times\\) 50,257

GPT-2 uses “tied” embedding weights: the same matrix is used for both embedding (a lookup) and unembed (a linear transformation) by simply transposing the matrix, i.e., \\(W\_U \= W\_E^T\\). This is, arguably, “[wildly unprincipled](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=BGVUas1SO2S3BqhVhBXw8wQl),” but reduces the model’s parameters by tens of millions.

### Decoding

Decoding is the process of selecting a single token given the output logits. Deterministic algorithms such as greedy selection (argmax) and varieties of beam search are possible, but stochastic sampling generally yields better results:

* Top-k filtering: Restrict selection to the k tokens with the highest logit values, setting remainder to –\\(\\infty\\).  
* Temperature scaling: Fine-tune the randomness of the sampling by scaling the logits prior to softmax. \\(T\<1\\) yields sharper probability distributions (closer to argmax), \\(T\>1\\) yields softer, more “creative” distributions.  
* Output softmax: Normalize the (filtered, scaled) logits to a probability distribution across the entire vocabulary.  
* Top-p filtering: Restrict selection to the highest probability tokens with cumulative probabilities adding to threshold p. (a/k/a nucleus sampling, cumulative probability mass)

Selection algorithms may also apply penalties based on the presence or frequency of a token already in the text.

## Terminology notes

* GPT-2: “GPT-2 124M,” the smallest model in [the GPT-2 series](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (and the most common reference model).  
* Token: One of \\(n\_{vocab}\\) elements which the tokenizer pre-processor decomposes the input text into. NB: There are no tokens inside of the model itself, only embeddings.  
* Position: the index value of a token from the input sequence, or the corresponding residual stream vector  
* Embedding: A dense, high-dimensional vector representing a token within the model. Lives in the embedding space; flows through the residual stream.   
* Layer: one “block” of a transformer, composed of an attention sublayer and an MLP sublayer.  
* \\(d\_{model}\\): The dimensionality of the token embeddings and the residual stream.  
* \\(d\_{head}\\): The dimensionality of the Query, Key, and Value vectors inside each attention head

