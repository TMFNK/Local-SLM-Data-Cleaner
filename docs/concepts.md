# Concepts, a deeper look

The README shows you how to run the project. This page explains the ideas behind
it, for readers who want to understand why it is built the way it is. You do not
need any of this to use the tool, but it helps if you want to adapt it.

## What a language model actually does

Strip away the mystique and a language model does one thing: given some text, it
predicts what text comes next. It reads its input as a sequence of tokens (chunks
of a few characters, roughly word pieces), and for each next position it computes
how likely every possible token is, then picks one. Do that repeatedly and you
get a sentence, a paragraph, or in our case a JSON record.

Everything the model "knows" lives in its parameters, the hundreds of millions of
numbers that turn input tokens into those predictions. Training is the process of
adjusting the numbers; running the model just uses them, which is why running is
cheap and can happen offline on a laptop.

One setting worth knowing: temperature. It controls how adventurous the
token-picking is. High temperature gives variety, which is nice for creative
writing and terrible for data cleaning. This project always calls the model with
temperature 0, meaning it always picks the most likely token, so the same input
gives the same output every time. Deterministic, repeatable, testable.

## Why a tiny model is enough

Big general models are trained to do almost anything: write code, translate poems,
argue philosophy. That breadth is expensive, and most of it is wasted if all you
need is one narrow job done reliably.

Data normalization is a narrow job. The set of things the model has to know is
small and regular: a handful of field types, a fixed list of country and currency
codes, a few date and number formats. A 0.6B model has more than enough capacity
to learn that. Once it is fine-tuned, it often matches or beats a much larger
general model on this specific task, because it has been shaped for exactly this
and nothing else.

The practical payoff is size. A 0.6B model runs in about 1 GB of memory, starts
instantly, and needs no GPU. You can run it on a laptop, embed it in a pipeline, or
ship it to a machine that never touches the internet.

## Base models vs instruct models

An open model like Qwen3 usually comes in two flavours.

A base model has only been trained to predict the next word on a huge pile of text.
It knows a lot about the world but does not follow instructions. Ask it a question
and it might continue the question rather than answer it.

An instruct model is a base model that went through an extra round of training to
follow instructions and hold a conversation. This is the version most people mean
when they say "an LLM".

For this project we fine-tune the instruct model. There is a real tradeoff here:

- The instruct model already knows how to follow a prompt and produce JSON, so it
  needs fewer training examples and gives you a usable zero-shot baseline before
  you train anything. That baseline is what makes the before-and-after comparison
  honest.
- A base model is a purer starting point with no built-in chat behaviour to work
  around, but it needs more data and more care to reach the same place, and it
  cannot give you a zero-shot baseline because it does nothing useful until trained.

Since our data is free to generate, either can work. We default to instruct because
it is simpler and the baseline story is cleaner. If you want to compare them, train
both on the same dataset and look at the two learning curves. That comparison is a
genuinely interesting result on its own.

## What fine-tuning changes, and what LoRA is

A model is a large pile of numbers called parameters. Training adjusts those
numbers so that, given an input, the model tends to produce the output you want.

Full fine-tuning would adjust all 600 million parameters. That is slow and memory
hungry. LoRA (Low-Rank Adaptation) takes a shortcut: it freezes the original
parameters and trains a small set of extra numbers layered on top. You end up
changing well under 1% of the model, which is why the whole thing runs on a laptop
in minutes. When you are done, `fuse` folds those extra numbers back into the model
so it can be exported and served as a single file.

That layering explains the `adapters/` folder the training step produces. The
base model stays untouched in its cache; the adapter is a small separate file
holding only the learned changes. You could keep several adapters for different
tasks on top of one base model. We fuse ours because we want a single
self-contained file to ship and serve.

The important mental shift: the model is not storing your examples in a lookup
table. It is adjusting its internal behaviour so the pattern generalizes. That is
why it can clean a record it never saw during training.

## Reading the training output

While `make train` runs, it prints a stream of lines with two numbers worth
understanding.

The train loss measures how wrong the model currently is on the examples it is
learning from. Each training step nudges the parameters in whatever direction
would have reduced that wrongness, so across the run the number falls: fast at
first, when the model is still learning the basic shape of the task, then more
slowly as it works on the details. The absolute value means little on its own.
The trend is the signal.

Every so often the trainer also prints a val loss, measured on the validation
split the model never trains on. This is the honesty check. If train loss keeps
falling while val loss stalls or rises, the model has started memorising its
exercises instead of learning the pattern, which is called overfitting. If you
crank the iterations way up or shrink the dataset, val loss is the number that
will tell you when to stop.

## Learning from a rule-based teacher (distillation)

Knowledge distillation normally means using a big expensive model as a teacher to
train a small cheap one. Here we do something slightly unusual: the teacher is not
a model at all, it is a deterministic algorithm (`normalize_record` in
`convention_spec.py`).

The algorithm knows the house rules exactly. We use it to label every training
example, which gives us two rare luxuries. The labels are always correct, and we
can make as many as we want for free. Most real-world fine-tuning projects spend
most of their effort getting clean labelled data. Here that problem disappears.

## How the synthetic data is made

The generator (`synth/generate.py`) works backwards from perfection. For each
example it first invents a fully clean, convention-valid record: a made-up vendor
in a made-up city with an invented IBAN, VAT number and phone number. Since we
built it clean, we know the right answer before we start.

Then it corrupts the record on purpose, using the same kinds of damage real
master data accumulates: padding and doubled spaces, `Germany` or `deutschland`
instead of `DE`, a euro sign instead of `EUR`, `stk` instead of `PCE`, IBANs
chopped into spaced groups, German decimal commas, dates flipped to `01.03.2024`,
and the occasional field replaced by `n/a` or `-` where a null belongs. Each
corruption fires randomly, so every record gets a different mix of problems.

Finally the deterministic algorithm cleans the corrupted record, and that output
becomes the label. The pair is checked at generation time, so a bad example
cannot enter the dataset. One detail matters for honesty: the corruptions are
random but the vocabulary of damage is finite, which is why the eval keeps a
held-out test split. It proves the model generalizes across records, not that we
tested it on the training file.

## Why bother with a model if the rules already work

This is the fair question, and the honest answer is about the long tail.

The algorithm handles every case someone thought to encode. Real data is full of
cases nobody encoded: an unusual abbreviation, a typo in a country name, a legal
form written three different ways, a date format from a legacy system. Each one is
another rule to write and maintain, forever.

A fine-tuned model learns the general shape of "normalize this", so it makes a
sensible attempt at inputs the rules never mentioned. It will not always be right,
which is why the runtime keeps the algorithm as a safety net and flags low
confidence output for review. The model widens coverage. The rules keep the known
cases exact. They work better together than either does alone.

## Quantization and the GGUF file

Model parameters are usually stored as 16-bit numbers. Quantization stores them
with fewer bits, for example 8-bit or 4-bit. The file gets smaller and runs faster,
at a small cost in precision.

You can see this directly in the files Step 8 produces. The first conversion
writes `qwen3-0.6b-cleaner.gguf` at full 16-bit precision, roughly 1.2 GB. The
quantize step then writes `qwen3-0.6b-cleaner-q8_0.gguf` at 8 bits per number,
around 600 MB, and that smaller file is the one the server runs.

For a 0.6B model the advice is to not over-compress. The memory you save going from
8-bit to 4-bit is tiny at this size, and the quality drop hurts a task that cares
about exact output. Q8_0 or Q6_K is a good balance.

GGUF is the file format `llama.cpp` uses. It packs the quantized weights plus the
tokenizer and settings into one file you can serve directly. It is an inference
format, which is why you cannot train it: fine-tuning happens on the original
weights, and you convert to GGUF only at the end.

## Why a server, and what "serving" means

Loading a model into memory takes a moment; answering with one is fast. A server
splits those apart. `llama-server` loads the GGUF file once, keeps it in memory,
and then listens on a local port (8080 by default) for requests. The eval and
demo scripts are just small programs that send a record to that port and read the
answer back.

The protocol on that port is the same OpenAI-style chat API that cloud LLMs
speak, which is deliberate. Any tool or library that can talk to a cloud model
can talk to your local one by pointing it at `http://localhost:8080` instead.
Nothing about the traffic leaves your machine; "localhost" is your Mac talking
to itself.

This is also why the guide keeps telling you to stop the server before training.
The model sitting in server memory and the training run both want the same RAM,
and on an 8 GB machine there is not room for both at once.

## Why the output is always valid JSON

A model left to its own devices will occasionally produce broken JSON: a missing
brace, a trailing comma, a hallucinated field. We prevent that with grammar
constrained decoding. `llama.cpp` can take a JSON schema and only allow the model
to produce tokens that keep the output valid against it.

This separates two different questions. Is the output well-formed JSON, which the
grammar guarantees, and is the content correct, which is what the model and the
eval are actually about. It means our accuracy numbers measure meaning, not
punctuation.

## How much data, and reading the learning curve

Because data is free, the question is not how much you can get but where more stops
helping. Train on a few sizes, for example 250, 500, 1000, 2000 examples, and plot
the eval accuracy for each. The curve rises steeply and then flattens. Stop near
the flat part. Adding data past that point mostly costs training time.

Quality still matters more than raw volume. A dataset that covers many corruption
types and field combinations teaches more than a larger dataset that repeats the
same easy cases. Because the algorithm validates every generated pair, our dataset
is filtered by construction.

## Limits and where this goes next

This v1 does one thing: normalize a single record to the convention. It does not
yet decide whether two records are the same company (deduplication), and it does
not reshape arbitrary nested JSON into the target schema. Those are the v2 and v3
items on the roadmap, and they are harder because they involve judgement and
structure rather than field-by-field rules.

It is also worth being clear about what a small local model is not. It is not a
general reasoning engine, and you should not point it at tasks far outside what it
was trained on. Its strength is being narrow, private, cheap, and predictable. For
the job it was built for, that is exactly what you want.
